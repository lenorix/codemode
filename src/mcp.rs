//! The MCP client: the production `ToolSource`, backed by `rmcp` over stdio
//! child processes. Lives on the engine island (connections are `!Send`);
//! connects lazily and reuses the connection.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use rmcp::model::CallToolRequestParams;
use rmcp::service::{Peer, RoleClient, RunningService, ServiceExt};
use rmcp::transport::TokioChildProcess;
use serde_json::Value;
use tokio::process::Command;

use crate::config::ServerConfig;
use crate::error::{Error, Result};
use crate::source::{LocalFuture, ToolSource};
use crate::types::{ServerInfo, ToolInfo};

struct Launch {
    name: String,
    command: String,
    args: Vec<String>,
    env: Vec<(String, String)>,
}

pub(crate) struct McpClient {
    launch: Vec<Launch>,
    connections: RefCell<HashMap<String, RunningService<RoleClient, ()>>>,
    /// Serializes the connect path so two concurrent first-uses of the same
    /// server don't each spawn a child process (the second would overwrite and
    /// orphan the first). Connects are rare (once per server), so one shared lock
    /// is enough and simpler than a per-server cell.
    connect_lock: tokio::sync::Mutex<()>,
    /// Secret-ish config (env values, command, args) redacted from error strings
    /// as best-effort defense-in-depth; the real control is that none of it leaves
    /// the child process. See [`McpClient::scrub`].
    secrets: Vec<String>,
}

impl McpClient {
    pub(crate) fn new(servers: &[ServerConfig]) -> Self {
        let launch: Vec<Launch> = servers
            .iter()
            .map(|s| Launch {
                name: s.name.clone(),
                command: s.command.clone(),
                args: s.args.clone(),
                env: s.env.clone(),
            })
            .collect();
        // Redact configured env values, command, and args, since tokens are commonly
        // passed as an arg (`--token SECRET`) or an env value. Only non-trivial
        // values are worth redacting; very short ones would over-redact (e.g. a
        // value of "1" appearing incidentally).
        let secrets = launch
            .iter()
            .flat_map(|l| {
                l.env
                    .iter()
                    .map(|(_, v)| v.clone())
                    .chain(std::iter::once(l.command.clone()))
                    .chain(l.args.iter().cloned())
            })
            .filter(|v| v.len() >= 4)
            .collect();
        Self {
            launch,
            connections: RefCell::new(HashMap::new()),
            connect_lock: tokio::sync::Mutex::new(()),
            secrets,
        }
    }

    /// Replace any configured secret value with `***`, so it never appears in an
    /// error surfaced to the model or written to the operator's logs.
    fn scrub(&self, mut message: String) -> String {
        for secret in &self.secrets {
            if message.contains(secret.as_str()) {
                message = message.replace(secret.as_str(), "***");
            }
        }
        message
    }

    /// A cloned `Peer` for `server`, connecting (and caching) on first use.
    async fn peer(&self, server: &str) -> Result<Peer<RoleClient>> {
        if let Some(service) = self.connections.borrow().get(server) {
            return Ok(service.peer().clone());
        }
        // Hold the connect lock and re-check: another task may have connected
        // this server while we awaited the lock.
        let _guard = self.connect_lock.lock().await;
        if let Some(service) = self.connections.borrow().get(server) {
            return Ok(service.peer().clone());
        }
        let cfg = self
            .launch
            .iter()
            .find(|l| l.name == server)
            .ok_or_else(|| Error::UnknownServer(server.to_string()))?;

        let mut command = Command::new(&cfg.command);
        command.args(&cfg.args);
        for (key, value) in &cfg.env {
            command.env(key, value);
        }
        let transport = TokioChildProcess::new(command).map_err(|e| Error::Spawn {
            server: server.to_string(),
            message: self.scrub(e.to_string()),
        })?;
        let service = ().serve(transport).await.map_err(|e| Error::Spawn {
            server: server.to_string(),
            message: self.scrub(e.to_string()),
        })?;
        let peer = service.peer().clone();
        self.connections
            .borrow_mut()
            .insert(server.to_string(), service);
        Ok(peer)
    }
}

impl ToolSource for McpClient {
    fn server_names(&self) -> Vec<String> {
        self.launch.iter().map(|l| l.name.clone()).collect()
    }

    fn servers(self: Rc<Self>) -> LocalFuture<Result<Vec<ServerInfo>>> {
        Box::pin(async move {
            Ok(self
                .launch
                .iter()
                .map(|l| ServerInfo {
                    name: l.name.clone(),
                    description: None,
                })
                .collect())
        })
    }

    fn tools(self: Rc<Self>, server: String) -> LocalFuture<Result<Vec<ToolInfo>>> {
        Box::pin(async move {
            let peer = self.peer(&server).await?;
            let tools = peer
                .list_all_tools()
                .await
                .map_err(|e| Error::Mcp(self.scrub(e.to_string())))?;
            Ok(tools
                .into_iter()
                .map(|t| ToolInfo {
                    name: t.name.to_string(),
                    description: t.description.map(|c| c.to_string()),
                    input_schema: Value::Object((*t.input_schema).clone()),
                })
                .collect())
        })
    }

    fn call(
        self: Rc<Self>,
        server: String,
        tool: String,
        args: Value,
    ) -> LocalFuture<Result<Value>> {
        Box::pin(async move {
            let peer = self.peer(&server).await?;
            let arguments = match args {
                Value::Object(map) => map,
                _ => serde_json::Map::new(),
            };
            let params = CallToolRequestParams::new(tool).with_arguments(arguments);
            let result = peer
                .call_tool(params)
                .await
                .map_err(|e| Error::Mcp(self.scrub(e.to_string())))?;
            map_result(result)
        })
    }

    fn shutdown(self: Rc<Self>) -> LocalFuture<()> {
        Box::pin(async move {
            let connections = std::mem::take(&mut *self.connections.borrow_mut());
            for (_, service) in connections {
                let _ = service.cancel().await;
            }
        })
    }
}

/// Turn a tool result into JSON: prefer structured content, else the text
/// blocks; a tool that flagged `is_error` surfaces as an error.
fn map_result(result: rmcp::model::CallToolResult) -> Result<Value> {
    let is_error = result.is_error == Some(true);
    if let Some(structured) = result.structured_content {
        return if is_error {
            Err(Error::Mcp(structured.to_string()))
        } else {
            Ok(structured)
        };
    }
    let text = result
        .content
        .iter()
        .filter_map(|c| c.as_text().map(|t| t.text.clone()))
        .collect::<Vec<_>>()
        .join("\n");
    if is_error {
        Err(Error::Mcp(text))
    } else {
        Ok(Value::String(text))
    }
}

#[cfg(test)]
mod tests {
    use rmcp::model::{CallToolResult, Content};
    use serde_json::json;

    use super::{McpClient, map_result};
    use crate::config::ServerConfig;

    #[test]
    fn scrubs_configured_secrets_from_errors() {
        // Secrets arrive via env *and* via args (`--token SECRET`); both redacted.
        let server = ServerConfig::stdio("api", "run-server", ["--token", "arg-secret-token"])
            .env("API_TOKEN", "super-secret-token-value");
        let client = McpClient::new(&[server]);
        let scrubbed = client.scrub(
            "spawn failed: API_TOKEN=super-secret-token-value --token arg-secret-token rejected"
                .to_string(),
        );
        assert!(
            !scrubbed.contains("super-secret-token-value"),
            "env secret leaked: {scrubbed}"
        );
        assert!(
            !scrubbed.contains("arg-secret-token"),
            "arg secret leaked: {scrubbed}"
        );
        assert!(scrubbed.contains("***"));
    }

    #[test]
    fn prefers_structured_content() {
        let result = CallToolResult::structured(json!({ "answer": 42 }));
        assert_eq!(map_result(result).unwrap(), json!({ "answer": 42 }));
    }

    #[test]
    fn joins_text_content_when_no_structured() {
        let result = CallToolResult::success(vec![Content::text("hello"), Content::text("world")]);
        assert_eq!(map_result(result).unwrap(), json!("hello\nworld"));
    }

    #[test]
    fn is_error_surfaces_as_err() {
        let result = CallToolResult::error(vec![Content::text("boom")]);
        assert!(map_result(result).is_err());
    }
}
