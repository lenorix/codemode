//! The backend seam. A `CodeRuntime` runs model-written code against generated
//! per-server modules, reaching the outside only through the `Bridge`. This is
//! the one extension point; community backends (other engines/languages)
//! implement this trait. It is intentionally minimal and unstable until a second
//! backend validates it. The frozen invariant is the `Bridge` contract: pure
//! data in, pure data out.

use std::rc::Rc;

use serde_json::Value;

use crate::source::LocalFuture;
use crate::types::{Capabilities, Limits, Outcome, ServerTools};

/// The single door out of the sandbox. Constructed by the brain already closed
/// over the per-run allowlist and tool-call budget, so the runtime just calls
/// it. `(server, tool, args) -> json`.
pub type Bridge = Rc<dyn Fn(String, String, Value) -> LocalFuture<Result<Value, BridgeError>>>;

/// Why a bridge call failed, surfaced to the code as an exception.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum BridgeError {
    Denied { server: String, tool: String },
    BudgetExceeded,
    Call(String),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeError::Denied { server, tool } => {
                write!(f, "tool '{server}/{tool}' is not exposed for this run")
            }
            BridgeError::BudgetExceeded => write!(f, "tool-call budget exceeded"),
            BridgeError::Call(msg) => write!(f, "{msg}"),
        }
    }
}

/// Everything one execution needs: the model's `source`, the `servers` to
/// generate modules for, the `bridge` to reach them, and the `limits` to honour.
///
/// `#[non_exhaustive]`: build it with [`RunRequest::new`] so adding a field later
/// (e.g. a per-run id) is not a breaking change for out-of-tree backends.
#[non_exhaustive]
pub struct RunRequest {
    /// The model-written program to run.
    pub source: String,
    /// The servers exposed to this run; the backend generates one module each.
    pub servers: Vec<ServerTools>,
    /// The one door out of the sandbox (already allowlist- and budget-checked).
    pub bridge: Bridge,
    /// The resource limits to apply.
    pub limits: Limits,
}

impl RunRequest {
    /// Assemble a request for one execution.
    pub fn new(source: String, servers: Vec<ServerTools>, bridge: Bridge, limits: Limits) -> Self {
        Self {
            source,
            servers,
            bridge,
            limits,
        }
    }
}

/// A backend that runs model-written code. This is the one extension point:
/// community backends (other engines or languages) implement it. It is the
/// "executor seam", the analog of Cloudflare Code Mode's `Executor` interface
/// (`execute(code, fns) -> {result, error, logs}`): the brain hands it the
/// source and a bridge to the tools, and gets an [`Outcome`] back.
///
/// A minimal backend (ignores the servers, just returns a constant):
///
/// ```ignore
/// use codemode::{CodeRuntime, RunRequest, Capabilities, Outcome, LocalFuture};
/// use serde_json::json;
///
/// struct NullRuntime;
/// impl CodeRuntime for NullRuntime {
///     fn capabilities(&self) -> Capabilities {
///         Capabilities::new("none", "This backend ignores your code.")
///     }
///     fn run(&self, _request: RunRequest) -> LocalFuture<Outcome> {
///         Box::pin(async { Outcome::ok(json!(null), Vec::new()) })
///     }
/// }
/// ```
///
/// A real backend, in `run`: stand up a fresh context, generate one module per
/// `request.servers` entry (each tool calls `request.bridge(server, tool, args_json)`),
/// run `request.source`, honour `request.limits`, and convert the result to JSON.
/// See `src/boa.rs` (in-process) and `SubprocessRuntime` (out-of-process) for the
/// two reference implementations, and `tests/common/` for the conformance suite.
///
/// Contract:
/// - `run` must execute `request.source` in a **fresh, isolated context** each
///   call (no state may leak between executions) and reach the outside world
///   *only* through `request.bridge`. Generating the per-server modules and
///   enforcing limits is the backend's responsibility.
/// - It must never panic the caller: a thrown exception, a timeout, or a tripped
///   limit is reported inside the returned [`Outcome`] (see [`crate::ExecError`]),
///   not as a panic.
/// - `capabilities` drives the model-facing guidance and is read once at startup.
///
/// Execution is island-local by contract: [`Bridge`] is `Rc<dyn Fn>` and `run`
/// returns a [`LocalFuture`] (both `!Send`), because the engine runs on a single
/// dedicated thread (Boa's `Context` is `!Send`). A naturally-`Send` backend (a
/// V8 isolate pool, an out-of-process engine) must still conform to this `!Send`
/// shape; `SubprocessRuntime` shows it crossing a process boundary.
///
/// Stability: the trait is **0.x and unstable**. It may change until a second
/// independent backend validates it. The frozen invariant is the [`Bridge`]:
/// pure data in, pure data out.
pub trait CodeRuntime {
    /// What the backend can do and the guidance shown to the model.
    fn capabilities(&self) -> Capabilities;

    /// Run one execution to completion, returning its [`Outcome`].
    fn run(&self, request: RunRequest) -> LocalFuture<Outcome>;
}
