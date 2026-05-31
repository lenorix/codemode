//! codemode-mcp — run model-written code against MCP tools instead of exposing
//! every tool directly. See `docs/` for the design and `docs/spec/` for the
//! contracts.

mod boa;
mod codemode;
mod config;
mod error;
mod local;
mod mcp;
mod runtime;
mod source;
mod types;

#[cfg(feature = "rig")]
pub mod rig;

#[cfg(feature = "subprocess")]
pub mod subprocess;

pub use boa::Boa;
pub use codemode::{CodeMode, CodeModeBuilder};
pub use config::{Config, ServerConfig};
pub use error::{Error, Result};
pub use local::LocalTools;
#[cfg(feature = "subprocess")]
pub use subprocess::SubprocessRuntime;
pub use runtime::{Bridge, BridgeError, CodeRuntime, RunRequest};
pub use source::LocalFuture;
pub use types::{
    Capabilities, ExecError, Limits, Outcome, ServerInfo, ServerTools, ToolInfo,
};
