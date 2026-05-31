//! End-to-end test of Surface A: spawn the `codemode-mcp serve` binary, connect
//! to it as an MCP client (the host's role), and exercise the three tools. No
//! downstream server needed — `execute` runs pure JS — so this is deterministic
//! and needs no network.

#![cfg(feature = "cli")]

use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use rmcp::transport::TokioChildProcess;
use tokio::process::Command;

#[test]
fn serve_exposes_three_tools_and_executes_code() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    tokio::task::LocalSet::new().block_on(&rt, async {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_codemode-mcp"));
        cmd.arg("serve");
        let client =
            ().serve(TokioChildProcess::new(cmd).expect("spawn serve"))
                .await
                .expect("connect");

        let tools = client.list_all_tools().await.expect("list tools");
        let names: Vec<String> = tools.iter().map(|t| t.name.to_string()).collect();
        for expected in ["discover", "find", "execute"] {
            assert!(
                names.iter().any(|n| n == expected),
                "missing tool {expected}; got {names:?}"
            );
        }

        let result = client
            .call_tool(
                CallToolRequestParams::new("execute").with_arguments(
                    serde_json::json!({ "code": "export default 6 * 7;" })
                        .as_object()
                        .unwrap()
                        .clone(),
                ),
            )
            .await
            .expect("call execute");

        let structured = result.structured_content.expect("structured content");
        assert_eq!(structured["result"], serde_json::json!(42));
        assert!(structured["error"].is_null());

        client.cancel().await.expect("cancel");
    });
}
