//! Configuration: how to launch downstream MCP servers, in our own TOML or the
//! standard `mcpServers` JSON, plus the runtime limits.

use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::types::Limits;

/// How to launch and expose one stdio MCP server.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub(crate) name: String,
    pub(crate) command: String,
    pub(crate) args: Vec<String>,
    pub(crate) env: Vec<(String, String)>,
    /// Exposed tools. `None` exposes every tool the server lists.
    pub(crate) allow: Option<Vec<String>>,
}

impl ServerConfig {
    pub fn stdio(
        name: impl Into<String>,
        command: impl Into<String>,
        args: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            name: name.into(),
            command: command.into(),
            args: args.into_iter().map(Into::into).collect(),
            env: Vec::new(),
            allow: None,
        }
    }

    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    /// Restrict the exposed tools (deny-by-default within this server).
    pub fn allow(mut self, tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.allow = Some(tools.into_iter().map(Into::into).collect());
        self
    }
}

/// Servers plus runtime limits, loadable from a file.
#[derive(Clone, Debug)]
pub struct Config {
    pub servers: Vec<ServerConfig>,
    pub limits: Limits,
}

impl Config {
    /// Load from a path, choosing the format by extension (`.json`/`.mcp.json`
    /// → MCP JSON, otherwise TOML).
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let is_json = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("json"));
        if is_json {
            Self::from_mcp_json(path)
        } else {
            Self::from_toml(path)
        }
    }

    pub fn from_toml(path: impl AsRef<Path>) -> Result<Self> {
        let text = read(path)?;
        let raw: TomlConfig = toml::from_str(&text).map_err(|e| Error::Config(e.to_string()))?;
        Ok(raw.into())
    }

    /// Load the standard `{ "mcpServers": { ... } }` shape. It carries no
    /// allowlist, so every listed server exposes all its tools.
    pub fn from_mcp_json(path: impl AsRef<Path>) -> Result<Self> {
        let text = read(path)?;
        let raw: McpJson = serde_json::from_str(&text).map_err(|e| Error::Config(e.to_string()))?;
        let servers = raw
            .mcp_servers
            .into_iter()
            .map(|(name, s)| ServerConfig {
                name,
                command: s.command,
                args: s.args,
                env: s.env.into_iter().collect(),
                allow: None,
            })
            .collect();
        Ok(Config {
            servers,
            limits: Limits::default(),
        })
    }
}

fn read(path: impl AsRef<Path>) -> Result<String> {
    std::fs::read_to_string(path.as_ref())
        .map_err(|e| Error::Config(format!("{}: {e}", path.as_ref().display())))
}

#[derive(Deserialize)]
struct TomlConfig {
    #[serde(default)]
    servers: std::collections::BTreeMap<String, TomlServer>,
    #[serde(default)]
    runtime: Option<TomlRuntime>,
}

#[derive(Deserialize)]
struct TomlServer {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    allow: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct TomlRuntime {
    timeout_ms: Option<u64>,
    max_loop_iterations: Option<u64>,
    max_recursion_depth: Option<usize>,
    max_stack_size: Option<usize>,
    max_backtrace_frames: Option<usize>,
    max_tool_calls: Option<u32>,
    max_output_bytes: Option<usize>,
    per_call_timeout_ms: Option<u64>,
}

impl From<TomlConfig> for Config {
    fn from(raw: TomlConfig) -> Self {
        let servers = raw
            .servers
            .into_iter()
            .map(|(name, s)| ServerConfig {
                name,
                command: s.command,
                args: s.args,
                env: s.env.into_iter().collect(),
                allow: s.allow,
            })
            .collect();
        let mut limits = Limits::default();
        if let Some(rt) = raw.runtime {
            if let Some(ms) = rt.timeout_ms {
                limits.timeout = std::time::Duration::from_millis(ms);
            }
            if let Some(v) = rt.max_loop_iterations {
                limits.max_loop_iterations = v;
            }
            if let Some(v) = rt.max_recursion_depth {
                limits.max_recursion_depth = v;
            }
            if let Some(v) = rt.max_stack_size {
                limits.max_stack_size = v;
            }
            if let Some(v) = rt.max_backtrace_frames {
                limits.max_backtrace_frames = v;
            }
            if let Some(v) = rt.max_tool_calls {
                limits.max_tool_calls = v;
            }
            if let Some(v) = rt.max_output_bytes {
                limits.max_output_bytes = v;
            }
            if let Some(ms) = rt.per_call_timeout_ms {
                limits.per_call_timeout = std::time::Duration::from_millis(ms);
            }
        }
        Config { servers, limits }
    }
}

#[derive(Deserialize)]
struct McpJson {
    #[serde(rename = "mcpServers")]
    mcp_servers: std::collections::BTreeMap<String, McpJsonServer>,
}

#[derive(Deserialize)]
struct McpJsonServer {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: std::collections::BTreeMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_toml_with_runtime_and_allow() {
        let toml = r#"
            [servers.gd]
            command = "npx"
            args = ["-y", "server-gdrive"]
            env = { TOKEN = "x" }
            allow = ["getSheet"]

            [runtime]
            timeout_ms = 2000
            max_tool_calls = 10
            max_backtrace_frames = 5
        "#;
        let cfg: Config = toml::from_str::<TomlConfig>(toml).unwrap().into();
        assert_eq!(cfg.servers.len(), 1);
        let s = &cfg.servers[0];
        assert_eq!(s.name, "gd");
        assert_eq!(s.command, "npx");
        assert_eq!(cfg.limits.max_backtrace_frames, 5);
        assert_eq!(s.allow.as_deref(), Some(&["getSheet".to_string()][..]));
        assert_eq!(cfg.limits.timeout, std::time::Duration::from_millis(2000));
        assert_eq!(cfg.limits.max_tool_calls, 10);
    }

    #[test]
    fn from_mcp_json_exposes_all_tools_with_default_limits() {
        let path = std::env::temp_dir().join("codemode_from_mcp_json_test.json");
        std::fs::write(
            &path,
            r#"{ "mcpServers": { "fs": { "command": "npx", "args": ["x"] } } }"#,
        )
        .unwrap();
        let cfg = Config::from_mcp_json(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(cfg.servers.len(), 1);
        assert_eq!(cfg.servers[0].name, "fs");
        assert!(
            cfg.servers[0].allow.is_none(),
            "mcp-json exposes all tools (no allowlist)"
        );
        assert_eq!(cfg.limits.timeout, Limits::default().timeout);
    }

    #[test]
    fn parses_mcp_json_with_allow_all() {
        let json = r#"{ "mcpServers": {
            "fs": { "command": "npx", "args": ["server-fs"], "env": { "K": "v" } }
        }}"#;
        let raw: McpJson = serde_json::from_str(json).unwrap();
        assert_eq!(raw.mcp_servers.len(), 1);
        let s = raw.mcp_servers.get("fs").unwrap();
        assert_eq!(s.command, "npx");
    }
}
