//! End-to-end test against a real MCP server (`@modelcontextprotocol/server-everything`).
//! Ignored by default: needs `npx` and network. Run with:
//!   cargo test --test integration_everything -- --ignored

use codemode_mcp::{CodeMode, ServerConfig};

#[tokio::test]
#[ignore = "requires npx + network"]
async fn end_to_end_against_server_everything() {
    let cm = CodeMode::builder()
        .server(ServerConfig::stdio(
            "everything",
            "npx",
            ["-y", "@modelcontextprotocol/server-everything"],
        ))
        .build()
        .await
        .expect("build");

    let servers = cm.discover().await.expect("discover");
    assert!(servers.iter().any(|s| s.name == "everything"));

    let tools = cm.find("everything").await.expect("find");
    assert!(
        tools.iter().any(|t| t.name == "echo"),
        "expected an `echo` tool, got: {:?}",
        tools.iter().map(|t| &t.name).collect::<Vec<_>>()
    );

    let outcome = cm
        .execute(
            "import * as e from './servers/everything';\n\
             const r = await e.echo({ message: 'hi from codemode' });\n\
             export default r;",
        )
        .await
        .expect("execute");

    assert!(
        outcome.error.is_none(),
        "execution error: {:?}",
        outcome.error
    );
    let rendered = serde_json::to_string(&outcome.result).unwrap();
    assert!(
        rendered.contains("hi from codemode"),
        "tool result did not echo the message: {rendered}"
    );

    cm.shutdown().await.expect("shutdown");
}
