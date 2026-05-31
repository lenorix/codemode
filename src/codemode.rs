//! The three-tool brain. `CodeMode` owns a single-threaded engine "island"
//! (its own current-thread tokio runtime + `LocalSet`) where the Boa context and
//! the MCP connections live, because both are `!Send`. Public methods marshal a
//! command to the island over a channel and await a oneshot reply, so `CodeMode`
//! itself is `Send + Sync`.

use std::cell::Cell;
use std::panic::AssertUnwindSafe;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use futures_util::FutureExt;
use serde_json::Value;
use tokio::sync::{OnceCell, Semaphore, mpsc, oneshot};

use crate::boa::Boa;
use crate::config::{Config, ServerConfig};
use crate::error::{Error, Result};
use crate::local::{CompositeSource, LocalTools};
use crate::mcp::McpClient;
use crate::runtime::{Bridge, BridgeError, CodeRuntime, RunRequest};
use crate::source::{AllowList, ToolSource};
use crate::types::{Capabilities, ExecError, Limits, Outcome, ServerInfo, ServerTools, ToolInfo};

/// Default ceiling on executions running at once on the island. Each running
/// execution holds a full JS context, so this bounds memory/CPU; further calls
/// queue until a slot frees. Override with [`CodeModeBuilder::max_concurrent_executions`].
const DEFAULT_MAX_CONCURRENT: usize = 16;

/// Builds the `ToolSource` on the island thread (MCP connections are `!Send`).
pub(crate) type SourceFactory = Box<dyn FnOnce() -> Rc<dyn ToolSource> + Send>;

enum Command {
    Discover(oneshot::Sender<Result<Vec<ServerInfo>>>),
    Find(String, oneshot::Sender<Result<Vec<ToolInfo>>>),
    Execute(String, oneshot::Sender<Outcome>),
    Shutdown(oneshot::Sender<()>),
}

/// A configured engine. The work happens on a background "island" thread.
pub struct CodeMode {
    tx: mpsc::UnboundedSender<Command>,
    capabilities: Capabilities,
    island: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for CodeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodeMode")
            .field("capabilities", &self.capabilities)
            .finish_non_exhaustive()
    }
}

impl CodeMode {
    /// Start building a `CodeMode`.
    pub fn builder() -> CodeModeBuilder {
        CodeModeBuilder::new()
    }

    /// Spawn the island. `runtime` and `factory` move onto the island thread.
    /// Fails only if the OS refuses the island's runtime or thread.
    pub(crate) fn spawn(
        factory: SourceFactory,
        runtime: Box<dyn CodeRuntime + Send>,
        allow: AllowList,
        limits: Limits,
        max_concurrent: usize,
    ) -> Result<Self> {
        let capabilities = runtime.capabilities();
        let (tx, mut rx) = mpsc::unbounded_channel::<Command>();

        // Build the island's runtime here so a failure is returned to the caller
        // rather than panicking the spawned thread.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Error::Island(e.to_string()))?;

        let max_concurrent = max_concurrent.max(1);
        let island = std::thread::Builder::new()
            .name("codemode-island".to_string())
            .spawn(move || {
                let local = tokio::task::LocalSet::new();
                local.block_on(&rt, async move {
                    let source = factory();
                    let runtime: Rc<dyn CodeRuntime + Send> = Rc::from(runtime);
                    let allow = Rc::new(allow);
                    let limits = Rc::new(limits);
                    // The exposed server set is fixed for a CodeMode's life. The
                    // OnceCell computes it once and single-flights it, so concurrent
                    // first executes don't each connect + list (racing the MCP cache).
                    let exposed = Rc::new(OnceCell::<Vec<ServerTools>>::new());
                    // Bounds executions running at once: each holds a JS context, so
                    // a flood of execute() calls queues instead of exhausting memory.
                    let slots = Arc::new(Semaphore::new(max_concurrent));
                    // Executions run concurrently: each is its own task with a fresh
                    // context (see boa.rs), so they never interfere. discover/find
                    // are cheap and run inline.
                    let mut tasks: Vec<tokio::task::JoinHandle<()>> = Vec::new();
                    while let Some(command) = rx.recv().await {
                        match command {
                            Command::Discover(reply) => {
                                let _ = reply.send(discover(&source, &allow).await);
                            }
                            Command::Find(server, reply) => {
                                let _ = reply.send(find(&source, &allow, &server).await);
                            }
                            Command::Execute(code, reply) => {
                                tasks.retain(|t| !t.is_finished());
                                let (source, runtime, allow, limits, exposed, slots) = (
                                    source.clone(),
                                    runtime.clone(),
                                    allow.clone(),
                                    limits.clone(),
                                    exposed.clone(),
                                    slots.clone(),
                                );
                                tasks.push(tokio::task::spawn_local(async move {
                                    // Wait for a free slot, then run. A closed
                                    // semaphore only happens at shutdown.
                                    let Ok(_permit) = slots.acquire_owned().await else {
                                        return;
                                    };
                                    // Catch a panic inside an execution so it becomes a
                                    // failed Outcome instead of dropping the reply (which
                                    // the caller would misread as a dead island).
                                    let run = AssertUnwindSafe(execute(
                                        &source,
                                        &allow,
                                        runtime.as_ref(),
                                        &limits,
                                        code,
                                        &exposed,
                                    ));
                                    let out = run.catch_unwind().await.unwrap_or_else(|_| {
                                        Outcome::failed(
                                            ExecError::Js {
                                                message: "internal error: execution panicked"
                                                    .to_string(),
                                            },
                                            Vec::new(),
                                        )
                                    });
                                    let _ = reply.send(out);
                                }));
                            }
                            Command::Shutdown(reply) => {
                                for task in tasks.drain(..) {
                                    let _ = task.await;
                                }
                                let _ = reply.send(());
                                break;
                            }
                        }
                    }
                    // Let any in-flight executions finish, then cancel MCP
                    // connections while the reactor is still alive.
                    for task in tasks.drain(..) {
                        let _ = task.await;
                    }
                    source.shutdown().await;
                });
            })
            .map_err(|e| Error::Island(e.to_string()))?;

        Ok(Self {
            tx,
            capabilities,
            island: Some(island),
        })
    }

    /// The engine's capabilities (language, async support, hard memory cap) and
    /// the LLM-facing usage guidance the surfaces put in front of the model.
    pub fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    /// List the servers exposed to the model's code.
    pub async fn discover(&self) -> Result<Vec<ServerInfo>> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Discover(tx))
            .map_err(|_| Error::IslandGone)?;
        rx.await.map_err(|_| Error::IslandGone)?
    }

    /// List a server's exposed tools and their input schemas.
    pub async fn find(&self, server: &str) -> Result<Vec<ToolInfo>> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Find(server.to_string(), tx))
            .map_err(|_| Error::IslandGone)?;
        rx.await.map_err(|_| Error::IslandGone)?
    }

    /// Run model-written code against the generated `./servers/<name>` modules.
    /// `Err` is a harness failure; a thrown exception, timeout, or tripped limit
    /// comes back in [`Outcome::error`].
    pub async fn execute(&self, code: &str) -> Result<Outcome> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Execute(code.to_string(), tx))
            .map_err(|_| Error::IslandGone)?;
        rx.await.map_err(|_| Error::IslandGone)
    }

    /// Stop the engine: drain in-flight executions, cancel MCP connections, and
    /// join the island thread.
    pub async fn shutdown(mut self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Shutdown(tx))
            .map_err(|_| Error::IslandGone)?;
        let _ = rx.await;
        if let Some(handle) = self.island.take() {
            let _ = handle.join();
        }
        Ok(())
    }
}

/// Fluent constructor for [`CodeMode`].
pub struct CodeModeBuilder {
    servers: Vec<ServerConfig>,
    local: Vec<LocalTools>,
    runtime: Box<dyn CodeRuntime + Send>,
    limits: Limits,
    max_concurrent: usize,
}

impl std::fmt::Debug for CodeModeBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodeModeBuilder")
            .field("servers", &self.servers)
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

impl CodeModeBuilder {
    fn new() -> Self {
        Self {
            servers: Vec::new(),
            local: Vec::new(),
            runtime: Box::new(Boa::new()),
            limits: Limits::default(),
            max_concurrent: DEFAULT_MAX_CONCURRENT,
        }
    }

    /// Cap how many executions run at once on the island (default 16). Further
    /// `execute` calls queue for a slot. Each running execution holds a JS
    /// context, so this bounds memory under a flood of calls.
    pub fn max_concurrent_executions(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }

    /// Expose a group of in-Rust tools as a virtual server (see [`LocalTools`]).
    pub fn local_tools(mut self, tools: LocalTools) -> Self {
        self.local.push(tools);
        self
    }

    /// Choose the engine. Defaults to [`Boa`].
    pub fn runtime(mut self, runtime: impl CodeRuntime + Send + 'static) -> Self {
        self.runtime = Box::new(runtime);
        self
    }

    pub fn server(mut self, server: ServerConfig) -> Self {
        self.servers.push(server);
        self
    }

    pub fn servers(mut self, servers: impl IntoIterator<Item = ServerConfig>) -> Self {
        self.servers.extend(servers);
        self
    }

    /// Add servers and limits from a loaded [`Config`].
    pub fn config(mut self, config: Config) -> Self {
        self.servers.extend(config.servers);
        self.limits = config.limits;
        self
    }

    pub fn limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.limits.timeout = timeout;
        self
    }

    pub async fn build(self) -> Result<CodeMode> {
        let mut allow = AllowList::default();
        for server in &self.servers {
            match &server.allow {
                Some(tools) => allow.allow(server.name.clone(), tools.clone()),
                None => allow.allow_all(server.name.clone()),
            }
        }
        for tools in &self.local {
            allow.allow(tools.server().to_string(), tools.tool_names());
        }

        let configs = self.servers;
        let locals = self.local;
        let factory: SourceFactory = Box::new(move || {
            let mut sources: Vec<Rc<dyn ToolSource>> = Vec::new();
            if !configs.is_empty() {
                sources.push(Rc::new(McpClient::new(&configs)));
            }
            for tools in locals {
                sources.push(tools.into_source());
            }
            Rc::new(CompositeSource::new(sources)) as Rc<dyn ToolSource>
        });
        CodeMode::spawn(
            factory,
            self.runtime,
            allow,
            self.limits,
            self.max_concurrent,
        )
    }
}

async fn discover(source: &Rc<dyn ToolSource>, allow: &AllowList) -> Result<Vec<ServerInfo>> {
    let servers = source.clone().servers().await?;
    Ok(servers
        .into_iter()
        .filter(|s| allow.server_allowed(&s.name))
        .collect())
}

async fn find(
    source: &Rc<dyn ToolSource>,
    allow: &AllowList,
    server: &str,
) -> Result<Vec<ToolInfo>> {
    if !allow.server_allowed(server) {
        return Err(Error::UnknownServer(server.to_string()));
    }
    let tools = source.clone().tools(server.to_string()).await?;
    Ok(tools
        .into_iter()
        .filter(|t| allow.allows(server, &t.name))
        .collect())
}

async fn execute(
    source: &Rc<dyn ToolSource>,
    allow: &AllowList,
    runtime: &dyn CodeRuntime,
    limits: &Limits,
    code: String,
    exposed: &OnceCell<Vec<ServerTools>>,
) -> Outcome {
    let servers = match exposed
        .get_or_try_init(|| exposed_servers(source, allow))
        .await
    {
        Ok(servers) => servers.clone(),
        Err(e) => {
            return Outcome::failed(
                crate::types::ExecError::Js {
                    message: e.to_string(),
                },
                Vec::new(),
            );
        }
    };
    let bridge = make_bridge(
        source.clone(),
        allow.clone(),
        limits.max_tool_calls,
        limits.per_call_timeout,
        limits.max_output_bytes,
    );
    runtime
        .run(RunRequest {
            source: code,
            servers,
            bridge,
            limits: limits.clone(),
        })
        .await
}

/// Gather the exposed servers and their allowed tools, for module codegen.
async fn exposed_servers(
    source: &Rc<dyn ToolSource>,
    allow: &AllowList,
) -> Result<Vec<ServerTools>> {
    let mut out = Vec::new();
    for server in source.clone().servers().await? {
        if !allow.server_allowed(&server.name) {
            continue;
        }
        // A server that fails to list its tools is skipped (not fatal): a broken
        // server shouldn't tear down executions that don't use it.
        let tools = match source.clone().tools(server.name.clone()).await {
            Ok(tools) => tools
                .into_iter()
                .filter(|t| allow.allows(&server.name, &t.name))
                .collect(),
            Err(e) => {
                eprintln!("codemode: skipping server '{}': {e}", server.name);
                continue;
            }
        };
        out.push(ServerTools {
            name: server.name,
            tools,
        });
    }
    Ok(out)
}

/// The bridge: the only door out of the sandbox. Enforces the allowlist and the
/// tool-call budget in Rust, applies a per-call timeout, caps the result size,
/// and scrubs internal error detail before it can reach the model.
fn make_bridge(
    source: Rc<dyn ToolSource>,
    allow: AllowList,
    max_calls: u32,
    per_call: Duration,
    max_output_bytes: usize,
) -> Bridge {
    let budget = Rc::new(Cell::new(0u32));
    Rc::new(
        move |server: String,
              tool: String,
              args: Value|
              -> crate::source::LocalFuture<std::result::Result<Value, BridgeError>> {
            let source = source.clone();
            let allow = allow.clone();
            let budget = budget.clone();
            Box::pin(async move {
                if !allow.allows(&server, &tool) {
                    return Err(BridgeError::Denied { server, tool });
                }
                let used = budget.get();
                if used >= max_calls {
                    return Err(BridgeError::BudgetExceeded);
                }
                budget.set(used + 1);

                // On timeout the call future is dropped mid-flight. That is safe on a
                // shared MCP connection: rmcp correlates responses by request id via a
                // per-request oneshot, so a late response to the cancelled call is
                // matched by id and discarded rather than desyncing the stream.
                let result =
                    match tokio::time::timeout(per_call, source.call(server, tool, args)).await {
                        // A local tool's own error message is intentional and safe to show.
                        Ok(Err(Error::Tool(message))) => return Err(BridgeError::Call(message)),
                        // MCP/spawn errors can carry paths or args; log the detail, surface a generic message.
                        Ok(Err(other)) => {
                            eprintln!("codemode: tool call failed: {other}");
                            return Err(BridgeError::Call("tool call failed".to_string()));
                        }
                        Err(_) => return Err(BridgeError::Call("tool call timed out".to_string())),
                        Ok(Ok(value)) => value,
                    };

                if !crate::types::serialized_within(&result, max_output_bytes) {
                    return Err(BridgeError::Call("tool result too large".to_string()));
                }
                Ok(result)
            })
        },
    )
}

/// Shared test fixtures: an in-memory `FakeSource` factory used by these tests
/// and the Rig adapter's tests.
#[cfg(test)]
pub(crate) mod test_support {
    use std::collections::HashMap;

    use serde_json::json;

    use super::*;
    use crate::source::fake::FakeSource;

    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    /// A `demo` server exposing `echo` and `secret`; the responder echoes the
    /// tool name and args.
    pub(crate) fn fake_factory() -> SourceFactory {
        fake_factory_counting(Arc::new(AtomicUsize::new(0)))
    }

    /// Like [`fake_factory`] but shares a counter of `tools()` calls, so tests
    /// can assert schema caching.
    pub(crate) fn fake_factory_counting(tools_calls: Arc<AtomicUsize>) -> SourceFactory {
        Box::new(move || {
            let mut tools = HashMap::new();
            tools.insert(
                "demo".to_string(),
                vec![
                    ToolInfo {
                        name: "echo".into(),
                        description: None,
                        input_schema: json!({}),
                    },
                    ToolInfo {
                        name: "secret".into(),
                        description: None,
                        input_schema: json!({}),
                    },
                ],
            );
            Rc::new(FakeSource {
                servers: vec![ServerInfo {
                    name: "demo".into(),
                    description: Some("Demo".into()),
                }],
                tools,
                responder: Box::new(|_server, tool, args| json!({ "tool": tool, "args": args })),
                tools_calls,
            }) as Rc<dyn ToolSource>
        })
    }

    /// A `demo` server plus a `broken` server that is listed but whose `tools()`
    /// errors (no entry in the map), to test graceful degradation.
    pub(crate) fn fake_factory_with_broken_server() -> SourceFactory {
        Box::new(|| {
            let mut tools = HashMap::new();
            tools.insert(
                "demo".to_string(),
                vec![ToolInfo {
                    name: "echo".into(),
                    description: None,
                    input_schema: json!({}),
                }],
            );
            Rc::new(FakeSource {
                servers: vec![
                    ServerInfo {
                        name: "demo".into(),
                        description: None,
                    },
                    ServerInfo {
                        name: "broken".into(),
                        description: None,
                    },
                ],
                tools,
                responder: Box::new(|_server, tool, args| json!({ "tool": tool, "args": args })),
                tools_calls: Arc::new(AtomicUsize::new(0)),
            }) as Rc<dyn ToolSource>
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::boa::Boa;

    fn code_mode(allow: AllowList) -> CodeMode {
        CodeMode::spawn(
            test_support::fake_factory(),
            Box::new(Boa::new()),
            allow,
            Limits::default(),
            DEFAULT_MAX_CONCURRENT,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn schema_cache_avoids_relisting() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let counter = Arc::new(AtomicUsize::new(0));
        let mut allow = AllowList::default();
        allow.allow("demo", ["echo"]);
        let cm = CodeMode::spawn(
            test_support::fake_factory_counting(counter.clone()),
            Box::new(Boa::new()),
            allow,
            Limits::default(),
            DEFAULT_MAX_CONCURRENT,
        )
        .unwrap();

        let code = "import * as d from './servers/demo'; export default await d.echo({});";
        cm.execute(code).await.unwrap();
        cm.execute(code).await.unwrap();
        cm.shutdown().await.unwrap();

        assert_eq!(
            counter.load(Ordering::Relaxed),
            1,
            "tools() should be listed once then cached"
        );
    }

    #[tokio::test]
    async fn discover_and_find_respect_the_allowlist() {
        let mut allow = AllowList::default();
        allow.allow("demo", ["echo"]); // secret is NOT exposed
        let cm = code_mode(allow);

        let servers = cm.discover().await.unwrap();
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].name, "demo");

        let tools = cm.find("demo").await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");

        cm.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn execute_calls_an_allowed_tool() {
        let mut allow = AllowList::default();
        allow.allow("demo", ["echo"]);
        let cm = code_mode(allow);

        let outcome = cm
            .execute(
                "import * as demo from './servers/demo';\n\
                 const r = await demo.echo({ n: 7 });\n\
                 export default r;",
            )
            .await
            .unwrap();

        assert!(outcome.error.is_none(), "error: {:?}", outcome.error);
        assert_eq!(
            outcome.result,
            json!({ "tool": "echo", "args": { "n": 7 } })
        );
        cm.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn executions_run_concurrently() {
        use std::sync::{Arc, Mutex};

        use crate::LocalTools;

        let order = Arc::new(Mutex::new(Vec::<String>::new()));
        let (o_slow, o_fast) = (order.clone(), order.clone());

        let cm = Arc::new(
            CodeMode::builder()
                .local_tools(
                    LocalTools::new("t")
                        .tool("slow", "", json!({ "type": "object" }), move |_| {
                            let o = o_slow.clone();
                            async move {
                                tokio::time::sleep(Duration::from_millis(120)).await;
                                o.lock().unwrap().push("slow".to_string());
                                Ok(json!(1))
                            }
                        })
                        .tool("fast", "", json!({ "type": "object" }), move |_| {
                            let o = o_fast.clone();
                            async move {
                                tokio::time::sleep(Duration::from_millis(10)).await;
                                o.lock().unwrap().push("fast".to_string());
                                Ok(json!(2))
                            }
                        }),
                )
                .build()
                .await
                .unwrap(),
        );

        let (cm1, cm2) = (cm.clone(), cm.clone());
        let (a, b) = tokio::join!(
            cm1.execute("import * as t from './servers/t'; export default await t.slow({});"),
            cm2.execute("import * as t from './servers/t'; export default await t.fast({});"),
        );
        assert!(a.unwrap().error.is_none());
        assert!(b.unwrap().error.is_none());

        // `slow` was submitted first but takes longer; if the two executions run
        // concurrently, `fast` finishes first.
        assert_eq!(
            *order.lock().unwrap(),
            vec!["fast".to_string(), "slow".to_string()]
        );
    }

    #[tokio::test]
    async fn slow_tool_times_out_and_island_survives() {
        use crate::LocalTools;

        let cm = CodeMode::builder()
            .limits(Limits {
                per_call_timeout: Duration::from_millis(50),
                ..Limits::default()
            })
            .local_tools(LocalTools::new("slow").tool(
                "wait",
                "Sleeps forever",
                json!({ "type": "object" }),
                |_args| async move {
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    Ok(json!("done"))
                },
            ))
            .build()
            .await
            .unwrap();

        let outcome = cm
            .execute("import * as s from './servers/slow'; export default await s.wait({});")
            .await
            .unwrap();
        assert!(
            outcome.error.is_some(),
            "expected the hung tool call to surface an error"
        );

        // The island survived the timeout: a subsequent execution still works.
        let after = cm.execute("export default 1 + 1;").await.unwrap();
        assert_eq!(after.result, json!(2));
        cm.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn oversized_tool_result_is_rejected() {
        use crate::LocalTools;

        let cm = CodeMode::builder()
            .limits(Limits {
                max_output_bytes: 100,
                ..Limits::default()
            })
            .local_tools(LocalTools::new("big").tool(
                "blob",
                "Returns a large value",
                json!({ "type": "object" }),
                |_args| async move { Ok(json!("x".repeat(10_000))) },
            ))
            .build()
            .await
            .unwrap();

        let outcome = cm
            .execute("import * as b from './servers/big'; export default await b.blob({});")
            .await
            .unwrap();
        assert!(
            outcome.error.is_some(),
            "oversized tool result should be rejected at the bridge"
        );
        cm.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn tool_call_budget_is_enforced() {
        use crate::LocalTools;

        let cm = CodeMode::builder()
            .limits(Limits {
                max_tool_calls: 2,
                ..Limits::default()
            })
            .local_tools(LocalTools::new("c").tool(
                "ping",
                "",
                json!({ "type": "object" }),
                |_args| async move { Ok(json!(1)) },
            ))
            .build()
            .await
            .unwrap();

        let outcome = cm
            .execute(
                "import * as c from './servers/c';\n\
                 for (let i = 0; i < 5; i++) await c.ping({});\n\
                 export default 'done';",
            )
            .await
            .unwrap();
        assert!(
            outcome.error.is_some(),
            "the 3rd call should exceed the budget"
        );
        cm.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn execute_calls_a_local_rust_tool() {
        use crate::LocalTools;

        let cm = CodeMode::builder()
            .local_tools(LocalTools::new("math").tool(
                "add",
                "Add two numbers",
                json!({ "type": "object" }),
                |args| async move {
                    let a = args["a"].as_f64().unwrap_or(0.0);
                    let b = args["b"].as_f64().unwrap_or(0.0);
                    Ok(json!(a + b))
                },
            ))
            .build()
            .await
            .unwrap();

        let outcome = cm
            .execute(
                "import * as math from './servers/math';\n\
                 export default await math.add({ a: 2, b: 3 });",
            )
            .await
            .unwrap();

        assert!(outcome.error.is_none(), "error: {:?}", outcome.error);
        assert_eq!(outcome.result, json!(5.0));
        cm.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn broken_server_does_not_break_execute() {
        let mut allow = AllowList::default();
        allow.allow("demo", ["echo"]);
        allow.allow_all("broken"); // exposed, but its tools() errors
        let cm = CodeMode::spawn(
            test_support::fake_factory_with_broken_server(),
            Box::new(Boa::new()),
            allow,
            Limits::default(),
            DEFAULT_MAX_CONCURRENT,
        )
        .unwrap();

        let outcome = cm
            .execute("import * as d from './servers/demo'; export default await d.echo({});")
            .await
            .unwrap();
        assert!(
            outcome.error.is_none(),
            "broken server should be skipped: {:?}",
            outcome.error
        );
        assert_eq!(outcome.result, json!({ "tool": "echo", "args": {} }));
        cm.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn invalid_direct_bridge_args_error_not_null() {
        let mut allow = AllowList::default();
        allow.allow("demo", ["echo"]);
        let cm = code_mode(allow);
        // Call the bridge directly with a non-JSON 3rd arg.
        let outcome = cm
            .execute(
                "const c = globalThis.__codemodeCall;\n\
                 await c('demo', 'echo', 'not json');\n\
                 export default 'unreached';",
            )
            .await
            .unwrap();
        match outcome.error {
            Some(crate::types::ExecError::Js { message }) => {
                assert!(message.contains("JSON"), "got: {message}");
            }
            other => panic!("expected a JS error, got {other:?}"),
        }
        cm.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn execute_denies_a_non_exposed_tool() {
        let mut allow = AllowList::default();
        allow.allow("demo", ["echo"]); // secret not allowed
        let cm = code_mode(allow);

        // Even calling the bridge directly must be denied in Rust.
        let outcome = cm
            .execute(
                "const c = globalThis.__codemodeCall;\n\
                 await c('demo','secret', JSON.stringify({}));\n\
                 export default 'unreached';",
            )
            .await
            .unwrap();

        match outcome.error {
            Some(crate::types::ExecError::Js { message }) => {
                assert!(message.contains("not exposed"), "got: {message}");
            }
            other => panic!("expected a JS error, got {other:?}"),
        }
        cm.shutdown().await.unwrap();
    }
}
