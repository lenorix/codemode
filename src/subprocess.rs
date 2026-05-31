//! Optional subprocess isolation (Unix): run the model's code in a child process
//! under OS resource limits (`RLIMIT_AS` for memory) plus a wall-clock kill, so a
//! runaway script can't take down the host — the hard CPU/memory bound that an
//! in-process engine can't give.
//!
//! `SubprocessRuntime` is just another [`CodeRuntime`]: instead of running Boa
//! in-process, it spawns a worker (`codemode-mcp __worker`) that runs Boa. The
//! child never holds secrets or connections; every tool call is proxied back to
//! the parent over a tiny line-delimited JSON protocol and serviced by the same
//! [`Bridge`] the in-process engine would use. The bridge is pure data
//! (`(server, tool, args) -> json`), which is exactly why it crosses the process
//! boundary unchanged.

use std::path::PathBuf;
use std::process::Stdio;
use std::rc::Rc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::boa::Boa;
use crate::runtime::{Bridge, BridgeError, CodeRuntime, RunRequest};
use crate::source::LocalFuture;
use crate::types::{Capabilities, ExecError, Limits, Outcome, ServerTools};

// --- The wire protocol (one JSON object per line) ---

/// The job the parent sends the worker (first line of the worker's stdin).
#[derive(Serialize, Deserialize)]
struct Job {
    source: String,
    servers: Vec<ServerTools>,
    limits: WireLimits,
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
    Call { server: String, tool: String, args: Value },
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
        Self { worker: worker.into(), max_memory_bytes: 512 * 1024 * 1024 }
    }

    /// Hard memory cap for the child (`RLIMIT_AS`, bytes). Default 512 MiB.
    pub fn max_memory_bytes(mut self, bytes: u64) -> Self {
        self.max_memory_bytes = bytes;
        self
    }
}

impl CodeRuntime for SubprocessRuntime {
    fn capabilities(&self) -> Capabilities {
        // Same JS engine, just isolated — identical guidance. RLIMIT_AS is a real
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

    #[cfg(unix)]
    {
        let mem = max_memory_bytes;
        // A coarse CPU-seconds cap as a second backstop to the wall-clock kill.
        let cpu_seconds = request.limits.timeout.as_secs().max(1) + 2;
        // Safety: only async-signal-safe calls (setrlimit) run between fork and exec.
        unsafe {
            command.pre_exec(move || set_resource_limits(mem, cpu_seconds));
        }
    }

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(e) => return failed(format!("could not spawn worker: {e}")),
    };
    let mut stdin = child.stdin.take().expect("piped stdin");
    let mut stdout = BufReader::new(child.stdout.take().expect("piped stdout")).lines();

    let job = Job {
        source: request.source,
        servers: request.servers,
        limits: WireLimits::from(&request.limits),
    };
    if let Err(e) = write_line(&mut stdin, &serde_json::to_string(&job).unwrap()).await {
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
    stdout: &mut tokio::io::Lines<BufReader<tokio::process::ChildStdout>>,
    stdin: &mut tokio::process::ChildStdin,
    bridge: &Bridge,
) -> Result<Outcome, String> {
    loop {
        let line = match stdout.next_line().await {
            Ok(Some(line)) => line,
            Ok(None) => {
                return Err("worker exited before producing a result (killed by a resource limit?)"
                    .to_string());
            }
            Err(e) => return Err(format!("reading from worker: {e}")),
        };
        match serde_json::from_str::<FromWorker>(&line).map_err(|e| e.to_string())? {
            FromWorker::Done { outcome } => return Ok(outcome),
            FromWorker::Call { server, tool, args } => {
                let reply = match bridge(server, tool, args).await {
                    Ok(value) => ToWorker::Ok { value },
                    Err(e) => ToWorker::Err { message: e.to_string() },
                };
                write_line(stdin, &serde_json::to_string(&reply).unwrap())
                    .await
                    .map_err(|e| format!("writing to worker: {e}"))?;
            }
        }
    }
}

fn failed(message: String) -> Outcome {
    Outcome::failed(ExecError::Js { message }, Vec::new())
}

async fn write_line<W: AsyncWriteExt + Unpin>(writer: &mut W, line: &str) -> std::io::Result<()> {
    writer.write_all(line.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await
}

#[cfg(unix)]
fn set_resource_limits(max_memory_bytes: u64, cpu_seconds: u64) -> std::io::Result<()> {
    // Best-effort: some platforms (notably macOS) don't honour RLIMIT_AS, and we
    // must not fail the spawn over it — the parent's wall-clock kill is the
    // cross-platform backstop. On Linux these are real hard caps.
    let mem = libc::rlimit {
        rlim_cur: max_memory_bytes as libc::rlim_t,
        rlim_max: max_memory_bytes as libc::rlim_t,
    };
    let cpu = libc::rlimit {
        rlim_cur: cpu_seconds as libc::rlim_t,
        rlim_max: (cpu_seconds + 1) as libc::rlim_t,
    };
    // Safety: setrlimit is async-signal-safe; we deliberately ignore failures.
    unsafe {
        libc::setrlimit(libc::RLIMIT_AS, &mem);
        libc::setrlimit(libc::RLIMIT_CPU, &cpu);
    }
    Ok(())
}

// --- Worker side (the child process) ---

struct WorkerIo {
    reader: BufReader<tokio::io::Stdin>,
    stdout: tokio::io::Stdout,
}

/// Entry point for the `__worker` subcommand. Reads one `Job`, runs Boa with a
/// bridge that proxies tool calls back to the parent, and writes the `Outcome`.
pub async fn run_worker() {
    let mut reader = BufReader::new(tokio::io::stdin());
    let mut line = String::new();
    if reader.read_line(&mut line).await.unwrap_or(0) == 0 {
        return; // no job
    }
    let job: Job = match serde_json::from_str(line.trim()) {
        Ok(job) => job,
        Err(_) => return,
    };

    let io = Rc::new(Mutex::new(WorkerIo { reader, stdout: tokio::io::stdout() }));
    let bridge = remote_bridge(io.clone());

    let outcome = Boa::new()
        .run(RunRequest {
            source: job.source,
            servers: job.servers,
            bridge,
            limits: job.limits.into(),
        })
        .await;

    let mut io = io.lock().await;
    let line = serde_json::to_string(&FromWorker::Done { outcome }).unwrap();
    let _ = write_line(&mut io.stdout, &line).await;
}

/// A bridge that turns each tool call into a round-trip with the parent. Calls
/// are serialized (one in flight at a time) so the protocol needs no request ids.
fn remote_bridge(io: Rc<Mutex<WorkerIo>>) -> Bridge {
    Rc::new(move |server: String, tool: String, args: Value| {
        let io = io.clone();
        Box::pin(async move {
            let mut io = io.lock().await;
            let call = serde_json::to_string(&FromWorker::Call { server, tool, args }).unwrap();
            write_line(&mut io.stdout, &call).await.map_err(|e| BridgeError::Call(e.to_string()))?;

            let mut line = String::new();
            let WorkerIo { reader, .. } = &mut *io;
            if reader.read_line(&mut line).await.map_err(|e| BridgeError::Call(e.to_string()))? == 0 {
                return Err(BridgeError::Call("parent closed the connection".to_string()));
            }
            match serde_json::from_str::<ToWorker>(line.trim())
                .map_err(|e| BridgeError::Call(e.to_string()))?
            {
                ToWorker::Ok { value } => Ok(value),
                ToWorker::Err { message } => Err(BridgeError::Call(message)),
            }
        }) as LocalFuture<_>
    })
}
