//! codemode-mcp: run model-written code against MCP tools instead of exposing
//! every tool directly. The model writes a small program that calls tools and
//! orchestrates them in a sandbox, so intermediate results never re-enter its
//! context. See `docs/` for the design and `docs/spec/` for the contracts.
//!
//! # The shape
//!
//! [`CodeMode`] is the engine. Build one with [`CodeMode::builder`], pointing it
//! at MCP servers ([`ServerConfig`]) and/or in-process tools ([`LocalTools`]).
//! It exposes exactly three primitives, discover (which servers), find (a
//! server's tools and schemas), and execute (run code against generated
//! `./servers/<name>` modules), and is `Send + Sync`.
//!
//! Two thin surfaces sit on top: the `codemode-mcp` binary (`serve`) is a stdio
//! MCP server any host configures, and the [`rig`] adapter (`.code_mode(&cm)`)
//! embeds it in a Rig agent. We never own the LLM loop.
//!
//! # Where to start reading
//!
//! - [`CodeMode`] / `src/codemode.rs`: the brain and the single-threaded engine
//!   "island" (Boa's `Context` and the MCP connections are `!Send`).
//! - [`CodeRuntime`] / `src/runtime.rs`: the one backend seam, plus the frozen
//!   [`Bridge`] invariant `(server, tool, args_json) -> Result<json>`.
//! - [`Boa`] / `src/boa.rs`: the default JavaScript backend.
//! - `tests/common/`: the conformance suite a new backend runs to behave like
//!   Boa (an integration-test harness, not shipped in the library).
//! - `src/mcp.rs`: the downstream MCP client; `src/source.rs`: the allowlist.

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

pub use boa::Boa;
pub use codemode::{CodeMode, CodeModeBuilder};
pub use config::{Config, ServerConfig};
pub use error::{Error, Result};
pub use local::LocalTools;
pub use runtime::{Bridge, BridgeError, CodeRuntime, RunRequest};
pub use source::LocalFuture;
pub use types::{Capabilities, ExecError, Limits, Outcome, ServerInfo, ServerTools, ToolInfo};
