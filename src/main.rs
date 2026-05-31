//! The `codemode-mcp` binary.
//!
//!   codemode-mcp serve [--config <file>]   run as a stdio MCP server (the main use)
//!   codemode-mcp tools [--config]          list the configured servers and their tools
//!
//! `serve` exposes exactly three tools (discover / find / execute) to a host, so
//! the host's model writes code instead of calling every downstream tool. The
//! downstream servers come from `--config` (our TOML or a standard `.mcp.json`).

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use codemode_mcp::{CodeMode, Config};
use rmcp::ServiceExt;
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ListToolsResult, PaginatedRequestParams,
    ServerCapabilities, ServerInfo, Tool,
};
use rmcp::ErrorData;
use rmcp::service::{RequestContext, RoleServer};
use rmcp::transport::stdio;
use serde_json::{Value, json};

#[derive(Parser)]
#[command(name = "codemode-mcp", version, about = "Run model-written code against MCP tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run as a stdio MCP server exposing discover/find/execute.
    Serve {
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// List the configured servers and their exposed tools (for inspection).
    Tools {
        #[arg(long)]
        config: Option<PathBuf>,
    },
}

fn main() -> ExitCode {
    // The hidden `__worker` subcommand is the child process of the subprocess
    // isolation runtime (see src/subprocess.rs). It's not a user-facing command.
    #[cfg(feature = "subprocess")]
    if std::env::args().nth(1).as_deref() == Some("__worker") {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("runtime");
        tokio::task::LocalSet::new().block_on(&rt, codemode_mcp::subprocess::run_worker());
        return ExitCode::SUCCESS;
    }

    let cli = Cli::parse();
    // A current-thread runtime + LocalSet: the MCP server transport uses
    // `spawn_local` (rmcp's `local` feature), so it must run on a LocalSet.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, run(cli))
}

async fn build_code_mode(config: Option<PathBuf>) -> Result<CodeMode, String> {
    let mut builder = CodeMode::builder();
    if let Some(path) = config {
        if path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("json")) {
            eprintln!(
                "warning: a .mcp.json carries no allowlist, so all listed tools are exposed; \
                 use a servers.toml with `allow = [...]` to restrict."
            );
        }
        let config = Config::from_path(&path).map_err(|e| e.to_string())?;
        builder = builder.config(config);
    }
    builder.build().await.map_err(|e| e.to_string())
}

async fn run(cli: Cli) -> ExitCode {
    match cli.command {
        Command::Serve { config } => serve(config).await,
        Command::Tools { config } => list_tools(config).await,
    }
}

async fn serve(config: Option<PathBuf>) -> ExitCode {
    let cm = match build_code_mode(config).await {
        Ok(cm) => Arc::new(cm),
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    // stdout is reserved for the MCP protocol; diagnostics go to stderr.
    let server = CodeModeServer { cm };
    let running = match server.serve(stdio()).await {
        Ok(running) => running,
        Err(e) => {
            eprintln!("error: failed to start server: {e}");
            return ExitCode::from(3);
        }
    };
    if let Err(e) = running.waiting().await {
        eprintln!("error: server stopped: {e}");
        return ExitCode::from(3);
    }
    ExitCode::SUCCESS
}

async fn list_tools(config: Option<PathBuf>) -> ExitCode {
    let cm = match build_code_mode(config).await {
        Ok(cm) => cm,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    let code = match cm.discover().await {
        Ok(servers) => {
            for server in &servers {
                println!("{}", server.name);
                match cm.find(&server.name).await {
                    Ok(tools) => {
                        for tool in tools {
                            let desc = tool.description.unwrap_or_default();
                            println!("  - {} {}", tool.name, desc);
                        }
                    }
                    Err(e) => println!("  (error listing tools: {e})"),
                }
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(3)
        }
    };
    let _ = cm.shutdown().await;
    code
}

/// The MCP server that fronts a `CodeMode`, exposing the three tools.
struct CodeModeServer {
    cm: Arc<CodeMode>,
}

fn schema(value: Value) -> Arc<rmcp::model::JsonObject> {
    Arc::new(value.as_object().cloned().unwrap_or_default())
}

impl ServerHandler for CodeModeServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.instructions = Some(self.cm.capabilities().usage_guidance.clone());
        info
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        let tools = vec![
            Tool::new(
                "discover",
                "List the MCP servers available to your code.",
                schema(json!({ "type": "object", "properties": {} })),
            ),
            Tool::new(
                "find",
                "List one server's tools and their input schemas.",
                schema(json!({
                    "type": "object",
                    "properties": { "server": { "type": "string" } },
                    "required": ["server"]
                })),
            ),
            Tool::new(
                "execute",
                self.cm.capabilities().usage_guidance.clone(),
                schema(json!({
                    "type": "object",
                    "properties": { "code": { "type": "string" } },
                    "required": ["code"]
                })),
            ),
        ];
        Ok(ListToolsResult::with_all_items(tools))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let args = request.arguments.unwrap_or_default();
        let value = match request.name.as_ref() {
            "discover" => {
                let servers = self.cm.discover().await.map_err(internal)?;
                serde_json::to_value(servers).map_err(internal)?
            }
            "find" => {
                let server = args
                    .get("server")
                    .and_then(Value::as_str)
                    .ok_or_else(|| ErrorData::invalid_params("missing 'server'", None))?;
                let tools = self.cm.find(server).await.map_err(internal)?;
                serde_json::to_value(tools).map_err(internal)?
            }
            "execute" => {
                let code = args
                    .get("code")
                    .and_then(Value::as_str)
                    .ok_or_else(|| ErrorData::invalid_params("missing 'code'", None))?;
                let outcome = self.cm.execute(code).await.map_err(internal)?;
                serde_json::to_value(outcome).map_err(internal)?
            }
            other => {
                return Err(ErrorData::invalid_params(format!("unknown tool '{other}'"), None));
            }
        };
        Ok(CallToolResult::structured(value))
    }
}

fn internal(e: impl std::fmt::Display) -> ErrorData {
    ErrorData::internal_error(e.to_string(), None)
}
