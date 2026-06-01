//! Optional subprocess isolation (Unix): run the model's code in a child process
//! under OS resource limits (`RLIMIT_AS` for memory, `RLIMIT_CPU`) plus a
//! wall-clock kill, so a runaway script can't take down the host. This is the
//! hard CPU/memory bound that an in-process engine can't give.
//!
//! `SubprocessRuntime` is just another [`CodeRuntime`]: instead of running Boa
//! in-process, it spawns a worker (`codemode-mcp __worker`) that runs Boa. The
//! child never holds secrets or connections; every tool call is proxied back to
//! the parent over a tiny line-delimited JSON protocol and serviced by the same
//! [`Bridge`] the in-process engine would use. The bridge is pure data
//! (`(server, tool, args) -> json`), which is exactly why it crosses the process
//! boundary unchanged.
//!
//! The worker caps its own address space and CPU (see `run_worker`) using the
//! safe `rlimit` wrapper, so there is no `unsafe` and no `pre_exec` here.

use std::io;
use std::path::PathBuf;
use std::process::Stdio;
use std::rc::Rc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::boa::Boa;
use crate::runtime::{Bridge, BridgeError, CodeRuntime, RunRequest};
use crate::source::LocalFuture;
use crate::types::{Capabilities, ExecError, Limits, Outcome, ServerTools};

/// Hard cap on a single protocol line. Payloads are already bounded by the
/// run's `max_output_bytes` (results) and by the worker's `RLIMIT_AS` (the JS
/// heap that builds tool-call arguments), but the *parent* is a separate process
/// not covered by the worker's rlimit, so it bounds incoming lines explicitly to
/// stop a giant tool-call argument from exhausting the parent's memory.
const MAX_LINE_BYTES: usize = 64 * 1024 * 1024;

// --- The wire protocol (one JSON object per line) ---

/// The job the parent sends the worker (first line of the worker's stdin). The
/// worker uses `max_memory_bytes` to cap its own address space (see `run_worker`),
/// which is why the cap travels in the job rather than being applied by the parent.
#[derive(Serialize, Deserialize)]
struct Job {
    source: String,
    servers: Vec<ServerTools>,
    limits: WireLimits,
    max_memory_bytes: u64,
}

/// `Limits` without `Duration` (which isn't plain serde), as milliseconds.
#[derive(Serialize, Deserialize)]
struct WireLimits {
    timeout_ms: u64,
    max_loop_iterations: u64,
    max_recursion_depth: usize,
    max_stack_size: usize,
    max_tool_calls: u32,
    max_output_bytes: usize,
    per_call_timeout_ms: u64,
}

impl From<&Limits> for WireLimits {
    fn from(l: &Limits) -> Self {
        Self {
            timeout_ms: l.timeout.as_millis() as u64,
            max_loop_iterations: l.max_loop_iterations,
            max_recursion_depth: l.max_recursion_depth,
            max_stack_size: l.max_stack_size,
            max_tool_calls: l.max_tool_calls,
            max_output_bytes: l.max_output_bytes,
            per_call_timeout_ms: l.per_call_timeout.as_millis() as u64,
        }
    }
}

impl From<WireLimits> for Limits {
    fn from(w: WireLimits) -> Self {
        Self {
            timeout: Duration::from_millis(w.timeout_ms),
            max_loop_iterations: w.max_loop_iterations,
            max_recursion_depth: w.max_recursion_depth,
            max_stack_size: w.max_stack_size,
            max_tool_calls: w.max_tool_calls,
            max_output_bytes: w.max_output_bytes,
            per_call_timeout: Duration::from_millis(w.per_call_timeout_ms),
        }
    }
}

/// Lines the worker writes to its stdout.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum FromWorker {
    /// A tool call the parent must service.
    Call {
        server: String,
        tool: String,
        args: Value,
    },
    /// The final result.
    Done { outcome: Outcome },
}

/// Lines the parent writes to the worker's stdin in response to a `Call`.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ToWorker {
    Ok { value: Value },
    Err { message: String },
}

// --- Parent side ---

/// A [`CodeRuntime`] that runs each execution in an isolated child process.
#[derive(Debug, Clone)]
pub struct SubprocessRuntime {
    worker: PathBuf,
    max_memory_bytes: u64,
}

impl SubprocessRuntime {
    /// Use the current executable as the worker (it must handle the `__worker`
    /// subcommand, which the `codemode-mcp` binary does).
    pub fn new() -> std::io::Result<Self> {
        Ok(Self::with_worker(std::env::current_exe()?))
    }

    /// Use a specific binary as the worker.
    pub fn with_worker(worker: impl Into<PathBuf>) -> Self {
        Self {
            worker: worker.into(),
            max_memory_bytes: 512 * 1024 * 1024,
        }
    }

    /// Hard memory cap for the child (`RLIMIT_AS`, bytes). Default 512 MiB.
    pub fn max_memory_bytes(mut self, bytes: u64) -> Self {
        self.max_memory_bytes = bytes;
        self
    }
}

impl CodeRuntime for SubprocessRuntime {
    fn capabilities(&self) -> Capabilities {
        // Same JS engine, just isolated, so identical guidance. RLIMIT_AS is a real
        // hard memory cap on Linux; macOS doesn't honour it (only the wall-clock
        // kill bounds a runaway there), so only claim the cap where it's true.
        let mut caps = Boa::new().capabilities();
        caps.hard_memory_cap = cfg!(target_os = "linux");
        caps
    }

    fn run(&self, request: RunRequest) -> LocalFuture<Outcome> {
        let worker = self.worker.clone();
        let max_memory_bytes = self.max_memory_bytes;
        Box::pin(async move { run_parent(worker, max_memory_bytes, request).await })
    }
}

async fn run_parent(worker: PathBuf, max_memory_bytes: u64, request: RunRequest) -> Outcome {
    let mut command = Command::new(&worker);
    command
        .arg("__worker")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(e) => return failed(format!("could not spawn worker: {e}")),
    };
    let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
        let _ = child.kill().await;
        return failed("worker stdio was not piped".to_string());
    };
    let mut stdin = stdin;
    let mut stdout = BufReader::new(stdout);

    let job = Job {
        source: request.source,
        servers: request.servers,
        limits: WireLimits::from(&request.limits),
        max_memory_bytes,
    };
    let job_line = match serde_json::to_string(&job) {
        Ok(line) => line,
        Err(e) => {
            let _ = child.kill().await;
            return failed(format!("could not encode job: {e}"));
        }
    };
    if let Err(e) = write_line(&mut stdin, &job_line).await {
        let _ = child.kill().await;
        return failed(format!("could not send job to worker: {e}"));
    }

    // Kill the child if it outlives the deadline (covers synchronous CPU, which
    // the in-process engine can't bound). Small margin over the run's own timeout.
    let deadline = request.limits.timeout + Duration::from_secs(2);
    let outcome = match tokio::time::timeout(
        deadline,
        pump(&mut stdout, &mut stdin, &request.bridge),
    )
    .await
    {
        Ok(Ok(outcome)) => outcome,
        Ok(Err(e)) => failed(e),
        Err(_) => Outcome::failed(ExecError::Timeout, Vec::new()),
    };
    let _ = child.kill().await;
    outcome
}

/// Service the worker's tool calls until it reports a result, or it exits early
/// (e.g. killed by a resource limit).
async fn pump(
    stdout: &mut BufReader<tokio::process::ChildStdout>,
    stdin: &mut tokio::process::ChildStdin,
    bridge: &Bridge,
) -> Result<Outcome, String> {
    loop {
        let line = match read_line_capped(stdout, MAX_LINE_BYTES).await {
            Ok(Some(line)) => line,
            Ok(None) => {
                return Err(
                    "worker exited before producing a result (killed by a resource limit?)"
                        .to_string(),
                );
            }
            Err(e) => return Err(format!("reading from worker: {e}")),
        };
        match serde_json::from_str::<FromWorker>(&line).map_err(|e| e.to_string())? {
            FromWorker::Done { outcome } => return Ok(outcome),
            FromWorker::Call { server, tool, args } => {
                let reply = match bridge(server, tool, args).await {
                    Ok(value) => ToWorker::Ok { value },
                    Err(e) => ToWorker::Err {
                        message: e.to_string(),
                    },
                };
                let line = serde_json::to_string(&reply).map_err(|e| e.to_string())?;
                write_line(stdin, &line)
                    .await
                    .map_err(|e| format!("writing to worker: {e}"))?;
            }
        }
    }
}

fn failed(message: String) -> Outcome {
    Outcome::failed(ExecError::Exception { message }, Vec::new())
}

async fn write_line<W: AsyncWriteExt + Unpin>(writer: &mut W, line: &str) -> std::io::Result<()> {
    writer.write_all(line.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await
}

/// Read one `\n`-delimited line, failing if it exceeds `max` bytes so a peer
/// can't force unbounded allocation. Every protocol message ends in `\n`, so EOF
/// before a newline means a truncated/absent message (e.g. the worker was killed
/// mid-write); that returns `Ok(None)`, which callers report as "the worker
/// exited" rather than as a confusing parse error on partial bytes.
async fn read_line_capped<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    max: usize,
) -> io::Result<Option<String>> {
    let too_long = || {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "protocol line exceeds maximum length",
        )
    };
    let mut buf: Vec<u8> = Vec::new();
    loop {
        let chunk = reader.fill_buf().await?;
        if chunk.is_empty() {
            return Ok(None); // EOF: no complete (newline-terminated) line
        }
        if let Some(pos) = chunk.iter().position(|&b| b == b'\n') {
            if buf.len() + pos > max {
                return Err(too_long());
            }
            buf.extend_from_slice(&chunk[..pos]);
            reader.consume(pos + 1);
            return Ok(Some(String::from_utf8_lossy(&buf).into_owned()));
        }
        let consumed = chunk.len();
        buf.extend_from_slice(chunk);
        reader.consume(consumed);
        if buf.len() > max {
            return Err(too_long());
        }
    }
}

/// Apply the child's own resource limits. Best-effort and deliberately ignoring
/// failures: the parent's wall-clock kill is the cross-platform backstop.
/// `RLIMIT_AS` is a real hard cap on Linux; other Unixes don't honour it (and
/// `rlimit` doesn't expose it there), so only CPU is set off Linux. This mirrors
/// `capabilities().hard_memory_cap` being Linux-only.
#[cfg(unix)]
fn apply_resource_limits(max_memory_bytes: u64, cpu_seconds: u64) {
    use rlimit::{Resource, setrlimit};
    let _ = setrlimit(Resource::CPU, cpu_seconds, cpu_seconds.saturating_add(1));
    #[cfg(target_os = "linux")]
    let _ = setrlimit(Resource::AS, max_memory_bytes, max_memory_bytes);
    #[cfg(not(target_os = "linux"))]
    let _ = max_memory_bytes;
}

// --- Worker side (the child process) ---

struct WorkerIo {
    reader: BufReader<tokio::io::Stdin>,
    stdout: tokio::io::Stdout,
}

/// Entry point for the `__worker` subcommand. Reads one `Job`, caps its own
/// resources, runs Boa with a bridge that proxies tool calls back to the parent,
/// and writes the `Outcome`.
pub async fn run_worker() {
    let mut reader = BufReader::new(tokio::io::stdin());
    let job: Job = match read_line_capped(&mut reader, MAX_LINE_BYTES).await {
        Ok(Some(line)) => match serde_json::from_str(line.trim()) {
            Ok(job) => job,
            Err(_) => return,
        },
        _ => return, // no job, or oversized/unreadable line
    };

    // Cap our own address space and CPU before running untrusted code. The CPU
    // cap is a coarse backstop just past the run's wall-clock timeout.
    #[cfg(unix)]
    apply_resource_limits(
        job.max_memory_bytes,
        (job.limits.timeout_ms / 1000).max(1).saturating_add(2),
    );

    let io = Rc::new(Mutex::new(WorkerIo {
        reader,
        stdout: tokio::io::stdout(),
    }));
    let bridge = remote_bridge(io.clone());

    let outcome = Boa::new()
        .run(RunRequest::new(
            job.source,
            job.servers,
            bridge,
            job.limits.into(),
        ))
        .await;

    let mut io = io.lock().await;
    if let Ok(line) = serde_json::to_string(&FromWorker::Done { outcome }) {
        let _ = write_line(&mut io.stdout, &line).await;
    }
}

/// A bridge that turns each tool call into a round-trip with the parent. Calls
/// are serialized (one in flight at a time) so the protocol needs no request ids.
fn remote_bridge(io: Rc<Mutex<WorkerIo>>) -> Bridge {
    Rc::new(move |server: String, tool: String, args: Value| {
        let io = io.clone();
        Box::pin(async move {
            let mut io = io.lock().await;
            let call = serde_json::to_string(&FromWorker::Call { server, tool, args })
                .map_err(|e| BridgeError::Call(e.to_string()))?;
            write_line(&mut io.stdout, &call)
                .await
                .map_err(|e| BridgeError::Call(e.to_string()))?;

            let WorkerIo { reader, .. } = &mut *io;
            let line = match read_line_capped(reader, MAX_LINE_BYTES).await {
                Ok(Some(line)) => line,
                Ok(None) => {
                    return Err(BridgeError::Call(
                        "parent closed the connection".to_string(),
                    ));
                }
                Err(e) => return Err(BridgeError::Call(e.to_string())),
            };
            match serde_json::from_str::<ToWorker>(line.trim())
                .map_err(|e| BridgeError::Call(e.to_string()))?
            {
                ToWorker::Ok { value } => Ok(value),
                ToWorker::Err { message } => Err(BridgeError::Call(message)),
            }
        }) as LocalFuture<_>
    })
}
