//! Subprocess isolation: the model's code runs in a child process with a hard
//! memory cap. Tool calls are proxied back to the parent. Unix + `subprocess`
//! feature only. Run: `cargo test --features subprocess --test subprocess`.

#![cfg(all(feature = "subprocess", unix))]

use codemode_mcp::{CodeMode, LocalTools, SubprocessRuntime};
use serde_json::json;

fn worker() -> SubprocessRuntime {
    // The test binary can't be the worker; use the built `codemode-mcp` binary,
    // which handles the hidden `__worker` subcommand.
    SubprocessRuntime::with_worker(env!("CARGO_BIN_EXE_codemode-mcp"))
}

#[tokio::test]
async fn runs_code_in_a_child_and_proxies_tool_calls() {
    let cm = CodeMode::builder()
        .runtime(worker())
        .local_tools(LocalTools::new("math").tool(
            "add",
            "Add two numbers",
            json!({ "type": "object" }),
            |args| async move {
                let a = args["a"].as_f64().unwrap_or(0.0);
                let b = args["b"].as_f64().unwrap_or(0.0);
                Ok(json!(a + b))
            },
        ))
        .build()
        .await
        .unwrap();

    let outcome = cm
        .execute(
            "import * as math from './servers/math';\n\
             export default await math.add({ a: 20, b: 22 });",
        )
        .await
        .unwrap();

    assert!(outcome.error.is_none(), "error: {:?}", outcome.error);
    assert_eq!(outcome.result, json!(42.0));
    cm.shutdown().await.unwrap();
}

// The same suite the in-process Boa backend passes, proving the seam holds
// across a process boundary for a second, independent `CodeRuntime`.
mod common;

#[test]
fn passes_the_conformance_suite() {
    common::assert_js_conformant(worker());
}

#[tokio::test]
async fn runaway_is_contained() {
    let cm = CodeMode::builder()
        .runtime(worker().max_memory_bytes(256 * 1024 * 1024))
        .build()
        .await
        .unwrap();

    // A huge allocation that the loop-iteration limit never sees. On Linux it
    // blows past RLIMIT_AS and the child is killed; elsewhere it's caught by the
    // output cap or the wall-clock kill. Either way the host stays up and the
    // execution surfaces an error rather than OOMing the parent.
    let outcome = cm
        .execute("export default 'x'.repeat(300000000);")
        .await
        .unwrap();

    assert!(
        outcome.error.is_some(),
        "expected the runaway to be contained, got {outcome:?}"
    );
    cm.shutdown().await.unwrap();
}
