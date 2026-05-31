//! Surface B: use codemode-mcp as a dependency inside a [Rig](https://docs.rig.rs)
//! agent — the `codemode()`-for-Rust shape. Instead of handing the agent every
//! MCP tool, `.code_mode(&cm)` gives it three tools (discover / find / execute)
//! and a preamble telling the model to write code.
//!
//! ```ignore
//! use std::sync::Arc;
//! use codemode_mcp::{CodeMode, ServerConfig};
//! use codemode_mcp::rig::CodeModeExt;
//! use rig::providers::anthropic;
//!
//! let cm = Arc::new(
//!     CodeMode::builder()
//!         .server(ServerConfig::stdio("fs", "npx", ["-y", "@modelcontextprotocol/server-filesystem", "."]))
//!         .build()
//!         .await?,
//! );
//!
//! let agent = anthropic::Client::from_env()
//!     .agent("claude-opus-4-8")
//!     .preamble("You are a helpful assistant.")
//!     .code_mode(&cm)
//!     .build();
//! ```

use std::sync::Arc;

// The external crate's lib name is `rig_core` (our own module is named `rig`).
use rig_core::agent::{AgentBuilder, NoToolConfig, PromptHook, WithBuilderTools};
use rig_core::completion::{CompletionModel, ToolDefinition};
use rig_core::tool::Tool;
use serde::Deserialize;

use crate::{CodeMode, Error, Outcome, ServerInfo, ToolInfo};

/// Extension on Rig's `AgentBuilder` that registers the three code-execution
/// tools and appends usage guidance derived from the runtime's capabilities.
pub trait CodeModeExt<M, P>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    fn code_mode(self, code_mode: &Arc<CodeMode>) -> AgentBuilder<M, P, WithBuilderTools>;
}

impl<M, P> CodeModeExt<M, P> for AgentBuilder<M, P, NoToolConfig>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    fn code_mode(self, cm: &Arc<CodeMode>) -> AgentBuilder<M, P, WithBuilderTools> {
        self.append_preamble(&cm.capabilities().usage_guidance)
            .tool(DiscoverTool(cm.clone()))
            .tool(FindTool(cm.clone()))
            .tool(ExecuteTool(cm.clone()))
    }
}

#[derive(Deserialize)]
struct NoArgs {}

#[derive(Deserialize)]
struct FindArgs {
    server: String,
}

#[derive(Deserialize)]
struct ExecuteArgs {
    code: String,
}

struct DiscoverTool(Arc<CodeMode>);
struct FindTool(Arc<CodeMode>);
struct ExecuteTool(Arc<CodeMode>);

impl Tool for DiscoverTool {
    const NAME: &'static str = "discover";
    type Error = Error;
    type Args = NoArgs;
    type Output = Vec<ServerInfo>;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "List the MCP servers available to your code.".to_string(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        }
    }

    async fn call(&self, _args: NoArgs) -> Result<Self::Output, Error> {
        self.0.discover().await
    }
}

impl Tool for FindTool {
    const NAME: &'static str = "find";
    type Error = Error;
    type Args = FindArgs;
    type Output = Vec<ToolInfo>;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "List one server's tools and their input schemas.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "server": { "type": "string" } },
                "required": ["server"]
            }),
        }
    }

    async fn call(&self, args: FindArgs) -> Result<Self::Output, Error> {
        self.0.find(&args.server).await
    }
}

impl Tool for ExecuteTool {
    const NAME: &'static str = "execute";
    type Error = Error;
    type Args = ExecuteArgs;
    type Output = Outcome;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            // Same guidance the MCP server uses, so the model sees one consistent
            // description (language + rules) however codemode is embedded.
            description: self.0.capabilities().usage_guidance.clone(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "code": { "type": "string" } },
                "required": ["code"]
            }),
        }
    }

    async fn call(&self, args: ExecuteArgs) -> Result<Self::Output, Error> {
        self.0.execute(&args.code).await
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::Boa;
    use crate::codemode::test_support::fake_factory;
    use crate::source::AllowList;
    use crate::types::Limits;

    fn fake_code_mode() -> Arc<CodeMode> {
        let mut allow = AllowList::default();
        allow.allow("demo", ["echo"]);
        Arc::new(CodeMode::spawn(fake_factory(), Box::new(Boa::new()), allow, Limits::default()))
    }

    #[tokio::test]
    async fn execute_tool_runs_code() {
        let tool = ExecuteTool(fake_code_mode());
        let outcome = tool
            .call(ExecuteArgs {
                code: "import * as demo from './servers/demo';\n\
                       export default await demo.echo({ n: 3 });"
                    .to_string(),
            })
            .await
            .unwrap();
        assert!(outcome.error.is_none(), "{:?}", outcome.error);
        assert_eq!(outcome.result, json!({ "tool": "echo", "args": { "n": 3 } }));
    }

    #[tokio::test]
    async fn discover_tool_lists_servers() {
        let tool = DiscoverTool(fake_code_mode());
        let servers = tool.call(NoArgs {}).await.unwrap();
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].name, "demo");
    }
}
