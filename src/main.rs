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
use codemode::{CodeMode, Config};
use rmcp::ErrorData;
use rmcp::ServiceExt;
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ListToolsResult, PaginatedRequestParams,
    ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::transport::stdio;
use serde_json::{Value, json};

#[derive(Parser)]
#[command(
    name = "codemode-mcp",
    version,
    about = "Run model-written code against MCP tools"
)]
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
        let rt = match build_runtime() {
            Ok(rt) => rt,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        };
        tokio::task::LocalSet::new().block_on(&rt, codemode::subprocess::run_worker());
        return ExitCode::SUCCESS;
    }

    let cli = Cli::parse();
    // A current-thread runtime + LocalSet: the MCP server transport uses
    // `spawn_local` (rmcp's `local` feature), so it must run on a LocalSet.
    let rt = match build_runtime() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, run(cli))
}

fn build_runtime() -> Result<tokio::runtime::Runtime, String> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("could not start the async runtime: {e}"))
}

async fn build_code_mode(config: Option<PathBuf>) -> Result<CodeMode, String> {
    let mut builder = CodeMode::builder();
    if let Some(path) = config {
        if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("json"))
        {
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
    // stdout is reserved for the MCP protocol; diagnostics go to stderr. Keep a
    // handle clone so we can tear down deterministically once the server stops.
    let server = CodeModeServer { cm: cm.clone() };
    let running = match server.serve(stdio()).await {
        Ok(running) => running,
        Err(e) => {
            eprintln!("error: failed to start server: {e}");
            return ExitCode::from(3);
        }
    };
    let result = running.waiting().await;
    // The server has stopped (the host disconnected), so `waiting` has dropped the
    // handler and its CodeMode handle. Reclaim sole ownership and shut down, so the
    // downstream MCP child processes are cancelled before we exit instead of being
    // orphaned by a racing process teardown (`list_tools` does the same). Best
    // effort: if a clone somehow outlives the server, skip the join rather than block.
    if let Ok(cm) = Arc::try_unwrap(cm) {
        let _ = cm.shutdown().await;
    }
    match result {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: server stopped: {e}");
            ExitCode::from(3)
        }
    }
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

/// The three tool definitions the server advertises. `execute_guidance` is the
/// runtime's usage guidance, surfaced as the `execute` tool's description.
/// Free-standing (no `RequestContext`) so it can be unit-tested directly.
fn tool_definitions(execute_guidance: &str) -> Vec<Tool> {
    vec![
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
            execute_guidance.to_string(),
            schema(json!({
                "type": "object",
                "properties": { "code": { "type": "string" } },
                "required": ["code"]
            })),
        ),
    ]
}

/// Run one of the three tools against `cm` and return its result as JSON. This is
/// the proxy's core: `discover`/`find`/`execute` map straight onto the `CodeMode`
/// methods. Kept free of the MCP `RequestContext` so it can be unit-tested without
/// standing up a transport.
async fn dispatch_tool(
    cm: &CodeMode,
    name: &str,
    args: &rmcp::model::JsonObject,
) -> Result<Value, ErrorData> {
    match name {
        "discover" => {
            let servers = cm.discover().await.map_err(internal)?;
            serde_json::to_value(servers).map_err(internal)
        }
        "find" => {
            let server = args
                .get("server")
                .and_then(Value::as_str)
                .ok_or_else(|| ErrorData::invalid_params("missing 'server'", None))?;
            let tools = cm.find(server).await.map_err(internal)?;
            serde_json::to_value(tools).map_err(internal)
        }
        "execute" => {
            let code = args
                .get("code")
                .and_then(Value::as_str)
                .ok_or_else(|| ErrorData::invalid_params("missing 'code'", None))?;
            let outcome = cm.execute(code).await.map_err(internal)?;
            serde_json::to_value(outcome).map_err(internal)
        }
        other => Err(ErrorData::invalid_params(
            format!("unknown tool '{other}'"),
            None,
        )),
    }
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
        let tools = tool_definitions(&self.cm.capabilities().usage_guidance);
        Ok(ListToolsResult::with_all_items(tools))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let args = request.arguments.unwrap_or_default();
        let value = dispatch_tool(&self.cm, request.name.as_ref(), &args).await?;
        Ok(CallToolResult::structured(value))
    }
}

fn internal(e: impl std::fmt::Display) -> ErrorData {
    ErrorData::internal_error(e.to_string(), None)
}

#[cfg(test)]
mod tests {
    //! Unit tests for the MCP-proxy surface that exercise the tool definitions
    //! and the discover/find/execute dispatch directly against a `CodeMode`,
    //! without spawning the binary or standing up an MCP transport. The
    //! end-to-end transport path is covered separately by `tests/serve.rs`.

    use super::*;
    use codemode::LocalTools;

    /// A `CodeMode` over one in-Rust "math" server with an `add` tool, so the
    /// dispatch path (including proxying a real tool call from `execute`) is
    /// deterministic and needs no downstream process or network.
    async fn math_cm() -> CodeMode {
        CodeMode::builder()
            .local_tools(LocalTools::new("math").tool(
                "add",
                "Add two numbers",
                json!({
                    "type": "object",
                    "properties": { "a": { "type": "number" }, "b": { "type": "number" } },
                    "required": ["a", "b"]
                }),
                |args| async move {
                    Ok(json!(
                        args["a"].as_f64().unwrap_or(0.0) + args["b"].as_f64().unwrap_or(0.0)
                    ))
                },
            ))
            .build()
            .await
            .unwrap()
    }

    fn args(value: Value) -> rmcp::model::JsonObject {
        value.as_object().cloned().unwrap_or_default()
    }

    #[test]
    fn tool_definitions_advertise_the_three_tools_with_guidance() {
        let defs = tool_definitions("GUIDANCE TEXT");
        let names: Vec<String> = defs.iter().map(|t| t.name.to_string()).collect();
        assert_eq!(names, ["discover", "find", "execute"]);
        let execute = defs.iter().find(|t| t.name == "execute").unwrap();
        assert_eq!(execute.description.as_deref(), Some("GUIDANCE TEXT"));
    }

    #[tokio::test]
    async fn dispatch_discover_lists_the_server() {
        let cm = math_cm().await;
        let value = dispatch_tool(&cm, "discover", &args(json!({})))
            .await
            .unwrap();
        assert_eq!(value[0]["name"], json!("math"));
        let _ = cm.shutdown().await;
    }

    #[tokio::test]
    async fn dispatch_find_lists_the_tools() {
        let cm = math_cm().await;
        let value = dispatch_tool(&cm, "find", &args(json!({ "server": "math" })))
            .await
            .unwrap();
        assert_eq!(value[0]["name"], json!("add"));
        let _ = cm.shutdown().await;
    }

    #[tokio::test]
    async fn dispatch_find_without_server_is_invalid_params() {
        let cm = math_cm().await;
        let err = dispatch_tool(&cm, "find", &args(json!({})))
            .await
            .unwrap_err();
        assert!(format!("{err:?}").contains("missing 'server'"), "{err:?}");
        let _ = cm.shutdown().await;
    }

    #[tokio::test]
    async fn dispatch_execute_runs_code_and_proxies_a_tool_call() {
        let cm = math_cm().await;
        let value = dispatch_tool(
            &cm,
            "execute",
            &args(json!({
                "code": "import * as math from './servers/math';\n\
                         export default await math.add({ a: 2, b: 3 });"
            })),
        )
        .await
        .unwrap();
        assert_eq!(value["result"], json!(5.0));
        assert!(value["error"].is_null(), "unexpected error: {value}");
        let _ = cm.shutdown().await;
    }

    #[tokio::test]
    async fn dispatch_execute_without_code_is_invalid_params() {
        let cm = math_cm().await;
        let err = dispatch_tool(&cm, "execute", &args(json!({})))
            .await
            .unwrap_err();
        assert!(format!("{err:?}").contains("missing 'code'"), "{err:?}");
        let _ = cm.shutdown().await;
    }

    #[tokio::test]
    async fn dispatch_unknown_tool_is_invalid_params() {
        let cm = math_cm().await;
        let err = dispatch_tool(&cm, "frobnicate", &args(json!({})))
            .await
            .unwrap_err();
        assert!(
            format!("{err:?}").contains("unknown tool 'frobnicate'"),
            "{err:?}"
        );
        let _ = cm.shutdown().await;
    }
}
