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
///
/// `#[non_exhaustive]`: construct it with [`Capabilities::new`] and set the
/// public fields you need, so adding a capability later is not a breaking change
/// for out-of-tree backends.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Capabilities {
    /// The language the backend runs (e.g. `"javascript"`). Shown to the model.
    pub language: &'static str,
    /// Whether the backend supports `await` / promises in model code.
    pub supports_async: bool,
    /// Whether the backend enforces a hard memory cap (true isolation), as
    /// opposed to compute-only sandboxing. Surfaces may relay this to operators.
    pub hard_memory_cap: bool,
    /// The LLM-facing rules the surfaces put in front of the model (how to import
    /// servers, how to return a value, language caveats). Never hardcoded by the
    /// agnostic layer; it comes from here.
    pub usage_guidance: String,
}

impl Capabilities {
    /// Capabilities for a backend running `language`, with `usage_guidance` shown
    /// to the model. `supports_async` and `hard_memory_cap` default to `false`;
    /// set those public fields afterwards if your backend provides them.
    pub fn new(language: &'static str, usage_guidance: impl Into<String>) -> Self {
        Self {
            language,
            supports_async: false,
            hard_memory_cap: false,
            usage_guidance: usage_guidance.into(),
        }
    }
}

/// Limits applied to one execution. `Default` is conservative and safe; start
/// from it and adjust fields. `#[non_exhaustive]` so adding a limit later does
/// not break callers (build with [`Limits::default`] rather than a literal).
///
/// These are advisory to the backend: a backend honours the limits it can and
/// may ignore those it can't express (see [`Capabilities`]).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Limits {
    /// Wall-clock deadline for the whole execution.
    pub timeout: Duration,
    /// Maximum loop iterations, for backends that can bound looping.
    pub max_loop_iterations: u64,
    /// Maximum recursion depth, for backends that can bound recursion.
    pub max_recursion_depth: usize,
    /// Max VM stack depth, for backends that bound recursion via a stack limit.
    pub max_stack_size: usize,
    /// Maximum number of tool calls one execution may make.
    pub max_tool_calls: u32,
    /// Maximum serialized size of a tool result or the final result.
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
            max_stack_size: 1024 * 10, // a typical engine default
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
    /// A successful run: the returned `result` and captured `logs`, no error.
    pub fn ok(result: serde_json::Value, logs: Vec<String>) -> Self {
        Self {
            result,
            logs,
            error: None,
        }
    }

    /// A failed run: the execution-level `error` and any `logs` captured before
    /// it, with a null result.
    pub fn failed(error: ExecError, logs: Vec<String>) -> Self {
        Self {
            result: serde_json::Value::Null,
            logs,
            error: Some(error),
        }
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

/// Max JSON nesting depth we'll convert or serialize. Both `serde_json`
/// serialization and `JsValue::from_json` recurse one native frame per level, so
/// a deeper value could overflow the stack and abort the process; this keeps
/// such conversions comfortably within the stack. Real JSON rarely nests past a
/// few dozen levels.
pub(crate) const MAX_JSON_DEPTH: usize = 256;

/// True if `value` nests no deeper than `max`, checked iteratively (so the check
/// itself can't overflow) and along the current path only (so a wide-but-shallow
/// value can't blow up memory). Used to vet a tool result before code that
/// recurses over it (the size-cap serializer, `JsValue::from_json`).
pub(crate) fn json_depth_within(value: &serde_json::Value, max: usize) -> bool {
    use serde_json::Value;
    enum Frame<'a> {
        Array(std::slice::Iter<'a, Value>),
        Object(serde_json::map::Values<'a>),
    }
    fn frame(value: &Value) -> Option<Frame<'_>> {
        match value {
            Value::Array(a) => Some(Frame::Array(a.iter())),
            Value::Object(o) => Some(Frame::Object(o.values())),
            _ => None,
        }
    }
    // `stack.len()` is the current nesting depth (the root container is depth 1).
    let mut stack: Vec<Frame> = frame(value).into_iter().collect();
    while !stack.is_empty() {
        let next = match stack.last_mut() {
            Some(Frame::Array(it)) => it.next(),
            Some(Frame::Object(it)) => it.next(),
            None => break,
        };
        match next {
            None => {
                stack.pop();
            }
            Some(child) => {
                if let Some(child_frame) = frame(child) {
                    if stack.len() + 1 > max {
                        return false;
                    }
                    stack.push(child_frame);
                }
            }
        }
    }
    true
}

/// An execution-level error, visible to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ExecError {
    /// An uncaught exception thrown by the code, or another execution-level
    /// error carrying a message (language-neutral; e.g. a JS `throw` in Boa).
    Exception { message: String },
    /// The wall-clock deadline elapsed.
    Timeout,
    /// A structural limit tripped (loop iterations, recursion, tool-call budget).
    Limit { what: String },
    /// Output exceeded `max_output_bytes`.
    OutputTooLarge,
}
