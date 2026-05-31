//! The library's error type.

/// Errors a `CodeMode` operation can return. These are harness failures
/// (config, spawning, transport). Errors *inside* the executed code — a thrown
/// exception, a timeout, a denied tool — are carried in [`crate::Outcome`]
/// instead, so the model can see them.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("unknown server: {0}")]
    UnknownServer(String),

    #[error("failed to start MCP server '{server}': {message}")]
    Spawn { server: String, message: String },

    #[error("MCP error: {0}")]
    Mcp(String),

    #[error("tool error: {0}")]
    Tool(String),

    #[error("the engine island stopped unexpectedly")]
    IslandGone,
}

pub type Result<T> = std::result::Result<T, Error>;
