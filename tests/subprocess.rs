//! Subprocess isolation: the model's code runs in a child process with a hard
//! memory cap. Tool calls are proxied back to the parent. Unix + `subprocess`
//! feature only. Run: `cargo test --features subprocess --test subprocess`.

#![cfg(all(feature = "subprocess", unix))]

use codemode::{CodeMode, ExecError, LocalTools, SubprocessRuntime};
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

#[tokio::test]
async fn concurrent_tool_calls_overlap() {
    use std::time::{Duration, Instant};

    // A tool that sleeps 300ms per call. We compare one call against three issued
    // together with Promise.all. With request-id pipelining the three overlap, so
    // they add ~no wall-clock over a single call; the old one-call-at-a-time worker
    // protocol would add ~600ms. Comparing the delta cancels the per-execution
    // worker cold-start (process spawn + Boa init), which a fixed bound can't.
    let cm = CodeMode::builder()
        .runtime(worker())
        .local_tools(LocalTools::new("t").tool(
            "slow",
            "Sleep 300ms then return 1",
            json!({ "type": "object" }),
            |_args| async move {
                tokio::time::sleep(Duration::from_millis(300)).await;
                Ok(json!(1))
            },
        ))
        .build()
        .await
        .unwrap();

    let started = Instant::now();
    let one = cm
        .execute("import * as t from './servers/t'; export default await t.slow({});")
        .await
        .unwrap();
    let one_elapsed = started.elapsed();
    assert!(one.error.is_none(), "error: {:?}", one.error);

    let started = Instant::now();
    let three = cm
        .execute(
            "import * as t from './servers/t';\n\
             const r = await Promise.all([t.slow({}), t.slow({}), t.slow({})]);\n\
             export default r.reduce((a, b) => a + b, 0);",
        )
        .await
        .unwrap();
    let three_elapsed = started.elapsed();
    assert!(three.error.is_none(), "error: {:?}", three.error);
    assert_eq!(three.result.as_f64(), Some(3.0));

    // Two extra concurrent 300ms calls must add well under one sleep's worth of
    // wall-clock (~0 when overlapping, ~600ms if serialized).
    let delta = three_elapsed.saturating_sub(one_elapsed);
    assert!(
        delta < Duration::from_millis(300),
        "concurrent tool calls did not overlap: one={one_elapsed:?} three={three_elapsed:?} delta={delta:?}"
    );
    cm.shutdown().await.unwrap();
}

#[tokio::test]
async fn deep_tool_result_is_rejected_cleanly_not_hung() {
    use std::time::{Duration, Instant};

    // A local tool that builds a value nested ~150 deep: above MAX_JSON_DEPTH, but
    // within the window that used to slip past the bridge and then fail to re-parse
    // in the worker, silently dropping the reply and hanging until the wall-clock
    // kill (~timeout + 2s). The bridge must now reject it up front with a clean
    // depth error, fast and identically to the in-process backend.
    let cm = CodeMode::builder()
        .runtime(worker())
        .local_tools(LocalTools::new("t").tool(
            "deep",
            "Return a deeply nested value",
            json!({ "type": "object" }),
            |_args| async move {
                let mut v = json!(0);
                for _ in 0..150 {
                    v = json!({ "n": v });
                }
                Ok(v)
            },
        ))
        .build()
        .await
        .unwrap();

    let started = Instant::now();
    let outcome = cm
        .execute("import * as t from './servers/t'; export default await t.deep({});")
        .await
        .unwrap();
    let elapsed = started.elapsed();

    assert!(
        matches!(&outcome.error, Some(ExecError::Exception { message }) if message.contains("deep")),
        "expected a clean depth error, got {:?}",
        outcome.error
    );
    assert!(
        elapsed < Duration::from_secs(3),
        "rejection should be fast (not a hang-until-kill), took {elapsed:?}"
    );
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
