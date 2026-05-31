//! Plain data that crosses the public boundary and the engine seam.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// A discovered server. Never carries launch config (command/args/env).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// A tool and its input schema, the source the codegen turns into a typed call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

/// One exposed server plus its tools, handed to a backend's codegen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerTools {
    pub name: String,
    pub tools: Vec<ToolInfo>,
}

/// What a backend can do and how the model should write code for it. The
/// surfaces render `usage_guidance` into the `execute` tool description.
#[derive(Debug, Clone)]
pub struct Capabilities {
    pub language: &'static str,
    pub supports_async: bool,
    pub hard_memory_cap: bool,
    pub usage_guidance: String,
}

/// Limits applied to one execution. `Default` is conservative and safe.
#[derive(Debug, Clone)]
pub struct Limits {
    pub timeout: Duration,
    pub max_loop_iterations: u64,
    pub max_recursion_depth: usize,
    /// Max VM stack depth (Boa's stack-size limit). Bounds non-tail recursion.
    pub max_stack_size: usize,
    pub max_tool_calls: u32,
    pub max_output_bytes: usize,
    /// Wall-clock timeout for a single tool call (independent of `timeout`).
    pub per_call_timeout: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(5),
            max_loop_iterations: 10_000_000,
            max_recursion_depth: 400,
            max_stack_size: 1024 * 10, // Boa's own default
            max_tool_calls: 50,
            max_output_bytes: 1_000_000,
            per_call_timeout: Duration::from_secs(30),
        }
    }
}

/// The result of running code: a returned value, captured console output, and
/// an optional execution-level error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    pub result: serde_json::Value,
    pub logs: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ExecError>,
}

impl Outcome {
    pub fn ok(result: serde_json::Value, logs: Vec<String>) -> Self {
        Self { result, logs, error: None }
    }

    pub fn failed(error: ExecError, logs: Vec<String>) -> Self {
        Self { result: serde_json::Value::Null, logs, error: Some(error) }
    }
}

/// True if `value` serializes to at most `max` bytes, without buffering it whole
/// (the sink aborts as soon as the budget is exceeded). Used to bound both tool
/// results coming in and the final result going out.
pub(crate) fn serialized_within(value: &serde_json::Value, max: usize) -> bool {
    struct Sink {
        written: usize,
        max: usize,
    }
    impl std::io::Write for Sink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.written += buf.len();
            if self.written > self.max {
                Err(std::io::Error::other("output too large"))
            } else {
                Ok(buf.len())
            }
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    serde_json::to_writer(Sink { written: 0, max }, value).is_ok()
}

/// An execution-level error, visible to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ExecError {
    /// A thrown or uncaught JavaScript error.
    Js { message: String },
    /// The wall-clock deadline elapsed.
    Timeout,
    /// A structural limit tripped (loop iterations, recursion, tool-call budget).
    Limit { what: String },
    /// Output exceeded `max_output_bytes`.
    OutputTooLarge,
}
