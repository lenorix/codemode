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

/// Everything one execution needs.
pub struct RunRequest {
    pub source: String,
    pub servers: Vec<ServerTools>,
    pub bridge: Bridge,
    pub limits: Limits,
}

pub trait CodeRuntime {
    fn capabilities(&self) -> Capabilities;
    fn run(&self, request: RunRequest) -> LocalFuture<Outcome>;
}
