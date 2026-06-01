//! The reusable, deterministic conformance suite for `CodeRuntime` backends.
//!
//! It lives here (a `tests/common/` integration-test module, shared by every
//! test file that declares `mod common;`) rather than in `src/`, because it is
//! testing code for backend authors, not production code. No LLM is involved: it
//! runs a fixed set of canonical programs and asserts language-neutral outcomes,
//! so a failure points squarely at the backend, not at a model.
//!
//! # The model: canonical program + backend translation
//!
//! The suite owns the *contract*, a fixed, named set of programs
//! ([`ConformanceProgram`]) and the language-neutral `Outcome` each must produce.
//! A backend owns the *translation*: it supplies the source for each program in
//! its own language. Anything language-specific lives in the translation, never
//! in the assertion (the isolation check, for instance, asserts the program
//! returns `true` when a global from the previous run is absent, and each backend
//! writes that "is it absent?" probe in its own syntax).
//!
//! A new backend added to this repo runs `assert_js_conformant(runtime)` (JS
//! backends) or `check(&runtime, its_translations)` with its own mapping.

use std::rc::Rc;

use codemode::{
    Bridge, CodeRuntime, ExecError, Limits, LocalFuture, Outcome, RunRequest, ServerTools, ToolInfo,
};
use serde_json::{Value, json};

/// One canonical program in the suite. A backend supplies the source for each in
/// its own language (see [`check`]); the suite owns the expected, language-neutral
/// outcome. Adding a variant is a deliberate change to the backend contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConformanceProgram {
    /// Return the number `42`.
    ReturnFortyTwo,
    /// Import the `demo` server and return `demo.echo({ n: 7 })` unchanged.
    EchoTool,
    /// Throw / raise an exception.
    Throw,
    /// Return `true` if no ambient host capabilities (fs/net/env/process) are
    /// reachable from the sandbox.
    NoHostGlobals,
    /// Loop forever (must be stopped by a limit or the deadline, not hang).
    InfiniteLoop,
    /// Recurse without a base case (must be stopped by a limit or the deadline).
    InfiniteRecursion,
    /// Return a value far larger than the output cap.
    HugeOutput,
    /// Set a global variable, then return anything. Runs before [`Self::ReadGlobal`].
    SetGlobal,
    /// Return `true` if the global set by [`Self::SetGlobal`] is absent this run.
    ReadGlobal,
}

impl ConformanceProgram {
    /// Every program, in the order [`check`] runs them. `SetGlobal` precedes
    /// `ReadGlobal` so the isolation check is meaningful.
    pub fn all() -> &'static [ConformanceProgram] {
        use ConformanceProgram::*;
        &[
            ReturnFortyTwo,
            EchoTool,
            Throw,
            NoHostGlobals,
            InfiniteLoop,
            InfiniteRecursion,
            HugeOutput,
            SetGlobal,
            ReadGlobal,
        ]
    }
}

/// Run the whole suite against `runtime`, taking each program's source from
/// `source`. Returns `Err(description)` on the first failed check (naming the
/// program and what went wrong), or `Ok(())` if all pass.
pub async fn check<R, F>(runtime: &R, source: F) -> Result<(), String>
where
    R: CodeRuntime,
    F: Fn(ConformanceProgram) -> String,
{
    for &program in ConformanceProgram::all() {
        run_one(runtime, program, &source(program)).await?;
    }
    Ok(())
}

/// The JavaScript translation of every [`ConformanceProgram`], for Boa and any
/// other JS backend to reuse directly.
pub fn javascript_programs() -> impl Fn(ConformanceProgram) -> String {
    use ConformanceProgram::*;
    |program| {
        match program {
            ReturnFortyTwo => "export default 6 * 7;",
            EchoTool => {
                "import * as demo from './servers/demo';\n\
                 export default await demo.echo({ n: 7 });"
            }
            Throw => "throw new Error('boom'); export default 1;",
            NoHostGlobals => {
                "export default typeof fetch === 'undefined' \
                 && typeof process === 'undefined' \
                 && typeof require === 'undefined' \
                 && typeof Deno === 'undefined';"
            }
            InfiniteLoop => "while (true) {} export default 1;",
            InfiniteRecursion => "function f() { return f(); } f(); export default 1;",
            // Comfortably over the 1 MiB default output cap.
            HugeOutput => "export default 'x'.repeat(2000000);",
            SetGlobal => "globalThis.__codemodeProbe = 123; export default 1;",
            ReadGlobal => "export default typeof globalThis.__codemodeProbe === 'undefined';",
        }
        .to_string()
    }
}

/// Run a JavaScript backend through the whole suite, panicking with a clear
/// message on the first failed check. Builds its own current-thread runtime +
/// `LocalSet` (a runtime's futures are `!Send`), so call it from a plain
/// `#[test]`, never from inside an existing async runtime.
pub fn assert_js_conformant<R: CodeRuntime>(runtime: R) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build runtime");
    tokio::task::LocalSet::new().block_on(&rt, async {
        check(&runtime, javascript_programs())
            .await
            .expect("backend should pass the conformance suite");
    });
}

async fn run_one<R: CodeRuntime>(
    runtime: &R,
    program: ConformanceProgram,
    src: &str,
) -> Result<(), String> {
    use ConformanceProgram::*;
    let name = format!("{program:?}");
    // Every program runs with the default limits; the canonical programs are
    // sized so the default loop and output caps are what trips them.
    let servers = if program == EchoTool {
        demo_servers()
    } else {
        vec![]
    };
    let out = run(runtime, src, servers).await;
    match program {
        ReturnFortyTwo => {
            expect_ok(&out, &name)?;
            expect_result(&out, &json!(42), &name)
        }
        EchoTool => {
            expect_ok(&out, &name)?;
            expect_result(&out, &json!({ "n": 7 }), &name)
        }
        Throw => expect_error(&out, &name, "a thrown exception", |e| {
            matches!(e, ExecError::Exception { .. })
        }),
        NoHostGlobals => {
            expect_ok(&out, &name)?;
            expect_result(&out, &json!(true), &name)
        }
        InfiniteLoop | InfiniteRecursion => expect_error(&out, &name, "Timeout or Limit", |e| {
            matches!(e, ExecError::Timeout | ExecError::Limit { .. })
        }),
        HugeOutput => expect_error(&out, &name, "OutputTooLarge", |e| {
            matches!(e, ExecError::OutputTooLarge)
        }),
        // Sets a global in its own (fresh) context; we only require it to run.
        SetGlobal => expect_ok(&out, &name),
        // A *different* fresh context: the global must not be visible, so the
        // backend's "is it absent?" probe returns true.
        ReadGlobal => {
            expect_ok(&out, &name)?;
            expect_result(&out, &json!(true), &name)
        }
    }
}

/// A bridge that echoes the call arguments back as the result.
fn echo_bridge() -> Bridge {
    Rc::new(|_server, _tool, args: Value| Box::pin(async move { Ok(args) }) as LocalFuture<_>)
}

/// A single "demo" server exposing an "echo" tool.
fn demo_servers() -> Vec<ServerTools> {
    vec![ServerTools {
        name: "demo".to_string(),
        tools: vec![ToolInfo {
            name: "echo".to_string(),
            description: None,
            input_schema: json!({ "type": "object" }),
        }],
    }]
}

async fn run<R: CodeRuntime>(runtime: &R, source: &str, servers: Vec<ServerTools>) -> Outcome {
    runtime
        .run(RunRequest::new(
            source.to_string(),
            servers,
            echo_bridge(),
            Limits::default(),
        ))
        .await
}

fn expect_ok(out: &Outcome, name: &str) -> Result<(), String> {
    match &out.error {
        None => Ok(()),
        Some(e) => Err(format!("{name}: expected success but got error {e:?}")),
    }
}

fn expect_result(out: &Outcome, want: &Value, name: &str) -> Result<(), String> {
    if &out.result == want {
        Ok(())
    } else {
        Err(format!(
            "{name}: expected result {want}, got {}",
            out.result
        ))
    }
}

fn expect_error(
    out: &Outcome,
    name: &str,
    want: &str,
    predicate: impl Fn(&ExecError) -> bool,
) -> Result<(), String> {
    match &out.error {
        Some(e) if predicate(e) => Ok(()),
        other => Err(format!("{name}: expected {want}, got {other:?}")),
    }
}
