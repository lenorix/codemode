//! The Boa backend: the first `CodeRuntime`. Runs model-written JavaScript on a
//! tokio current-thread event loop, exposes each MCP server as a generated
//! `./servers/<name>` module, and reaches tools only through the bridge.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ops::DerefMut;
use std::panic::AssertUnwindSafe;
use std::rc::Rc;
use std::time::Duration;

use futures_util::FutureExt;

use boa_engine::builtins::promise::PromiseState;
use boa_engine::context::ContextBuilder;
use boa_engine::context::time::JsInstant;
use boa_engine::job::{GenericJob, Job, JobExecutor, NativeAsyncJob, PromiseJob, TimeoutJob};
use boa_engine::module::{ModuleLoader, Referrer};
use boa_engine::prelude::{Finalize, JsData, Trace};
use boa_engine::property::PropertyKey;
use boa_engine::{
    Context, JsArgs, JsError, JsNativeError, JsObject, JsResult, JsString, JsValue, Module,
    NativeFunction, Source, js_string,
};

use crate::runtime::{Bridge, CodeRuntime, RunRequest};
use crate::source::LocalFuture;
use crate::types::{Capabilities, ExecError, Limits, Outcome, ServerTools, ToolInfo};

/// Per-execution state, attached to the Boa `Context` via host-defined data
/// (`insert_data`/`get_data`) rather than thread-locals. Each execution owns its
/// own state through its own context, so concurrent runs never share it. None of
/// the fields hold GC pointers, hence `unsafe_ignore_trace`.
#[derive(Trace, Finalize, JsData)]
struct RunState {
    #[unsafe_ignore_trace]
    bridge: Bridge,
    #[unsafe_ignore_trace]
    logs: RefCell<Vec<String>>,
    #[unsafe_ignore_trace]
    log_budget: Cell<usize>,
}

/// The Boa engine. Cheap to construct; a fresh context is built per execution.
#[derive(Debug, Default, Clone, Copy)]
pub struct Boa;

impl Boa {
    pub fn new() -> Self {
        silence_expected_limit_panic();
        Self
    }
}

/// Boa raises a Rust *panic* (not a catchable JS error) when a runtime limit
/// (loop, recursion, or stack) trips during module evaluation, because it can't
/// turn that error into a promise-rejection reason. `run_inner` catches that
/// panic and returns a clean [`ExecError::Limit`], so it's fully handled and
/// routine for a sandbox running untrusted code. The default panic hook would
/// still print Boa's confusing internal message to stderr on every limit hit, so
/// we install a process-global hook (once) that drops exactly that message and
/// forwards every other panic to the previous hook unchanged.
fn silence_expected_limit_panic() {
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let payload = info.payload();
            let message = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or_default();
            if message.contains("RuntimeLimit native error") {
                return; // expected; handled by run_inner's catch_unwind
            }
            previous(info);
        }));
    });
}

impl CodeRuntime for Boa {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            language: "javascript",
            supports_async: true,
            hard_memory_cap: false,
            usage_guidance: GUIDANCE.to_string(),
        }
    }

    fn run(&self, request: RunRequest) -> LocalFuture<Outcome> {
        Box::pin(async move {
            // A fresh Context is built (and dropped) inside `run_inner`, so all of
            // this execution's JS heap and per-run state is released when it
            // returns; nothing accumulates or leaks across runs.
            let (result, logs) = run_inner(
                &request.source,
                &request.servers,
                request.bridge,
                &request.limits,
            )
            .await;
            match result {
                Ok(value) => Outcome::ok(value, logs),
                Err(err) => Outcome::failed(err, logs),
            }
        })
    }
}

const GUIDANCE: &str = "\
Write one JavaScript ES module and return its answer with `export default <value>` \
at the top level. This is required: a `return`, `console.log`, or `module.exports` \
does not return anything, and `console.log` only adds debug lines to `logs`. \
Top-level await is allowed.

Reach tools only by importing per-server modules, exactly \
`import * as fs from './servers/filesystem'` (the `./servers/` prefix is required), \
then calling the exported functions. Use the exact tool names from `find`; each \
takes a single object argument (e.g. `await fs.readFile({ path })`) and returns a \
Promise. You have no network, no filesystem, and no `fetch` of your own; the \
imported server modules are the only way out.

Modern JavaScript works (optional chaining, spread, destructuring, template \
literals, Array/Object/Map/Set/JSON, Promise.all). `Intl` and `structuredClone` \
are not available; format numbers with `toFixed`. Do the whole task in this one \
program, and `export default` only the answer the user needs: filter, select, and \
aggregate inside the code rather than returning raw tool output, so only the \
final result returns.";

/// Build a fresh context, run `source`, and return `(result, captured logs)`.
/// Logs are read from the context's `RunState` afterwards, so they're returned
/// on every path (success, exception, timeout, limit).
async fn run_inner(
    source: &str,
    servers: &[ServerTools],
    bridge: Bridge,
    limits: &Limits,
) -> (Result<serde_json::Value, ExecError>, Vec<String>) {
    // Bound the source before parsing it. `boa_parser` has no recursion limit, so
    // deeply nested/recursive source would overflow this thread's stack and abort
    // the process. Capping the source size relative to the (roomy) island stack
    // makes that overflow unreachable for any accepted input (see
    // `check_source_size`). Runs no JS.
    if let Err(e) = check_source_size(source) {
        return (Err(e), Vec::new());
    }
    let loader = Rc::new(ServerModuleLoader::new(servers));
    let queue = Rc::new(Queue::default());
    let mut context = match ContextBuilder::new()
        .job_executor(queue.clone())
        .module_loader(loader)
        .build()
    {
        Ok(context) => context,
        Err(e) => return (Err(js(e, limits.max_output_bytes)), Vec::new()),
    };

    {
        let rl = context.runtime_limits_mut();
        rl.set_loop_iteration_limit(limits.max_loop_iterations);
        rl.set_recursion_limit(limits.max_recursion_depth);
        rl.set_stack_size_limit(limits.max_stack_size);
    }

    if let Err(e) = install_globals(&mut context) {
        return (Err(js(e, limits.max_output_bytes)), Vec::new());
    }
    context.insert_data(RunState {
        bridge,
        logs: RefCell::new(Vec::new()),
        log_budget: Cell::new(limits.max_output_bytes),
    });

    let result = async {
        let module = Module::parse(Source::from_bytes(source.as_bytes()), None, &mut context)
            .map_err(|e| js(e, limits.max_output_bytes))?;

        // Boa panics when a runtime limit (loop / recursion / stack) fires during
        // *module* evaluation, because it can't turn that error into a promise
        // rejection reason. We catch that panic and return a clean Limit error;
        // the context is fresh per run and dropped right after, so a poisoned
        // engine state can't leak.
        let eval = AssertUnwindSafe(async {
            let promise = module.load_link_evaluate(&mut context);
            queue
                .clone()
                .run_jobs_async(&RefCell::new(&mut context))
                .await?;
            Ok::<_, JsError>(promise)
        })
        .catch_unwind();

        let promise = match tokio::time::timeout(limits.timeout, eval).await {
            Err(_) => return Err(ExecError::Timeout),
            Ok(Err(_panic)) => {
                return Err(ExecError::Limit {
                    what:
                        "execution aborted: a runtime limit (loop, recursion or stack) was exceeded"
                            .to_string(),
                });
            }
            Ok(Ok(Err(err))) => return Err(classify(err.to_string(), limits.max_output_bytes)),
            Ok(Ok(Ok(promise))) => promise,
        };

        match promise.state() {
            PromiseState::Fulfilled(_) => {
                exported_value(&module, &mut context, limits.max_output_bytes)
            }
            // Render the rejection reason WITHOUT invoking user code. This match
            // runs after the wall-clock deadline, so calling `to_string` on an
            // attacker-thrown object would execute its `toString`/`valueOf`/
            // `Symbol.toPrimitive` here, unbounded past the timeout (the same
            // post-deadline hazard `result_convertible` guards on the fulfilled
            // path). `render_rejection` reads stored properties only.
            PromiseState::Rejected(err) => Err(classify(
                render_rejection(&err, &mut context),
                limits.max_output_bytes,
            )),
            PromiseState::Pending => Err(ExecError::Exception {
                message: "module did not finish executing".to_string(),
            }),
        }
    }
    .await;

    // Move the captured logs out (the RunState is discarded right after), rather
    // than cloning the whole Vec<String>.
    let logs = context
        .remove_data::<RunState>()
        .map(|state| std::mem::take(&mut *state.logs.borrow_mut()))
        .unwrap_or_default();
    (result, logs)
}

fn js(err: impl ToString, max: usize) -> ExecError {
    ExecError::Exception {
        message: cap_message(err.to_string(), max),
    }
}

/// Maximum source we'll hand the parser. This is the anti-abort guard for the
/// parser stack overflow, and it works by being calibrated against the parse
/// thread's stack rather than by inspecting the source. `boa_parser` is recursive
/// descent with no depth limit, so deeply nested or right-recursive source
/// (nested brackets, long `?:` / `=` / `**` / `new` chains) recurses one native
/// frame per source token and would overflow the stack and *abort* the process
/// during `Module::parse` (before evaluation, uncatchable by `catch_unwind`). The
/// densest case is a run of open brackets: one frame, about 20 KiB of stack, per
/// source byte. So if the source is no larger than `island_stack / 20 KiB`, even
/// an all-brackets source can't drive the parser past the stack: the overflow
/// becomes unreachable for *any* input within the cap, with no need to parse or
/// scan it (which we proved can't be done soundly).
///
/// The pairing is the invariant: this cap times ~20 KiB must stay safely under
/// [`crate::CodeMode`]'s island stack (8 KiB x 20 KiB = 160 MiB, comfortably under
/// the 256 MiB stack). Raise them together if a larger source is ever needed. The
/// parse runs on the single island thread, so it's one stack and there's no
/// concurrency multiplier on the worst-case transient use. Real model-written
/// orchestration code is far under 8 KiB.
const MAX_SOURCE_BYTES: usize = 8 * 1024;

/// Refuse a source larger than [`MAX_SOURCE_BYTES`] before we parse it, so the
/// parser can't be driven to a stack-overflow abort (see the constant's note).
fn check_source_size(source: &str) -> Result<(), ExecError> {
    if source.len() > MAX_SOURCE_BYTES {
        Err(ExecError::Limit {
            what: format!("source is larger than {MAX_SOURCE_BYTES} bytes"),
        })
    } else {
        Ok(())
    }
}

/// Extract a fulfilled module's `export default` as JSON, vetting it on the way
/// out. A missing export becomes a fix-naming error (the most common LLM mistake)
/// rather than a silent `null`; then the value is checked for the depth and
/// sparse-array hazards that would otherwise abort `to_json`, and for the size cap.
fn exported_value(
    module: &Module,
    context: &mut Context,
    max_output_bytes: usize,
) -> Result<serde_json::Value, ExecError> {
    let default = module
        .namespace(context)
        .get(js_string!("default"), context)
        .map_err(|e| js(e, max_output_bytes))?;
    if default.is_undefined() {
        return Err(ExecError::Exception {
            message: "no value was exported: assign your answer with \
                      `export default <value>` at the top level (not `return`, \
                      `console.log`, or `module.exports`)"
                .to_string(),
        });
    }
    // `to_json` recurses one native frame per nesting level (so a deep result
    // would overflow the island stack and *abort* the process), and it eagerly
    // `Vec::with_capacity(arr.length)`s arrays (so a sparse array with a huge
    // `.length` would try to allocate tens of GB and abort), both before the size
    // cap can see the result. Vet the value iteratively first.
    result_convertible(&default, context, MAX_RESULT_DEPTH, max_output_bytes)?;
    let value = default
        .to_json(context)
        .map_err(|e| js(e, max_output_bytes))?
        .unwrap_or(serde_json::Value::Null);
    check_output_size(&value, max_output_bytes)?;
    Ok(value)
}

/// Bound the serialized result without buffering it whole.
fn check_output_size(value: &serde_json::Value, max_bytes: usize) -> Result<(), ExecError> {
    if crate::types::serialized_within(value, max_bytes) {
        Ok(())
    } else {
        Err(ExecError::OutputTooLarge)
    }
}

/// Max nesting depth of an exported value, before `to_json` (which recurses one
/// native frame per level) can overflow the stack. Shared with the inbound
/// tool-result guard (see [`crate::types::MAX_JSON_DEPTH`]).
const MAX_RESULT_DEPTH: usize = crate::types::MAX_JSON_DEPTH;

/// Vet that `value` is safe to hand to `to_json`, without native recursion (so
/// the check itself can't overflow). Rejects two things `to_json` would turn
/// into a process abort before the size cap could see them: nesting deeper than
/// `max_depth`, and arrays whose `length` exceeds `max_array_len` (`to_json`
/// eagerly `Vec::with_capacity`s that length, so a sparse `a.length = 4e9` would
/// try to allocate tens of GB). An array longer than `max_array_len` can't fit
/// the output cap anyway (each element serializes to at least one byte), so a
/// `max_array_len` of `max_output_bytes` never rejects a result the size cap
/// would have accepted.
///
/// It runs NO user JavaScript. `to_json` reads stored data-property values
/// straight from the object's property map (substituting `null` for accessors)
/// and never invokes getters or Proxy traps, so we mirror that exactly: values
/// are read from the property descriptor, and Proxy objects (whose `ownKeys`
/// trap is user code) are not enumerated. This matters for more than fidelity.
/// This walk runs *after* the wall-clock deadline that wraps evaluation, so a
/// getter or trap firing here would execute unbounded code past the timeout.
///
/// It's a depth-first walk whose explicit stack holds only the *current path*
/// (one frame per level), so the stack height equals the current depth and a
/// wide level can never inflate it. Every node is visited and there's no
/// early-accept by shape, so a deep branch hidden behind wide padding can't slip
/// past. The number of visits is itself capped at `max_array_len`: a value that
/// fans out past that (a DAG of shared sub-objects, which `to_json` re-expands
/// once per reference) can't fit the output cap anyway, and the cap keeps both
/// this walk and the `to_json` that follows from spinning on it.
///
/// Best-effort, not a hard guarantee: other synchronous hostile compute (ReDoS,
/// a single huge allocation) isn't covered and can't be bounded in-process. It
/// cleanly handles the realistic cases; isolating deliberately hostile code is
/// the host's responsibility (see docs/security.md).
fn result_convertible(
    value: &JsValue,
    context: &mut Context,
    max_depth: usize,
    max_array_len: usize,
) -> Result<(), ExecError> {
    use boa_engine::builtins::proxy::Proxy;

    // The stored data value for `key`, or None for an accessor or missing
    // property (exactly what `to_json` recurses into). Reads the property map
    // directly, so it never fires a getter.
    let stored = |object: &JsObject, key: &PropertyKey| -> Option<JsValue> {
        object
            .borrow()
            .properties()
            .get(key)
            .and_then(|d| d.value().cloned())
    };
    // `to_json` eagerly `Vec::with_capacity`s an array's `length`, so a sparse
    // `a.length = 4e9` aborts. An array's `length` is always a data property
    // (the spec forbids an accessor there), so reading it fires no getter.
    let too_long = |object: &JsObject| -> bool {
        object.is_array()
            && stored(object, &js_string!("length").into())
                .and_then(|len| len.as_number())
                .is_some_and(|len| len > max_array_len as f64)
    };
    let too_wide = || ExecError::Limit {
        what: format!("exported array is longer than {max_array_len} elements"),
    };
    // Own keys to descend into. A Proxy's `ownKeys` trap is user code, and
    // `to_json` reads a Proxy's own storage as empty anyway, so we skip it.
    let keys = |object: &JsObject, context: &mut Context| -> Vec<PropertyKey> {
        if object.is::<Proxy>() {
            Vec::new()
        } else {
            object.own_property_keys(context).unwrap_or_default()
        }
    };

    let Some(root) = value.as_object() else {
        return Ok(());
    };
    if too_long(&root) {
        return Err(too_wide());
    }
    let root_keys = keys(&root, context);
    // Frame = (object, its keys, index of the next key to visit).
    let mut stack: Vec<(JsObject, Vec<PropertyKey>, usize)> = vec![(root, root_keys, 0)];
    // Bound the total work, not just the depth. `to_json` expands a shared
    // sub-object once per reference (its cycle guard is path-local), so a DAG
    // like `for (...) x = { l: x, r: x }` stays shallow yet serializes to
    // exponentially many elements; this walk mirrors that expansion, so the same
    // shape would spin here too. Both run after the wall-clock deadline on the
    // shared island thread, so an unbounded walk would hang every concurrent
    // execution. A value that serializes within the output cap emits at most
    // `max_array_len` elements (each is at least one byte), so a higher count
    // means the result is too large regardless; reject it as the size cap would.
    let mut visited = 0usize;
    loop {
        let Some(top) = stack.last() else {
            return Ok(());
        };
        if top.2 >= top.1.len() {
            stack.pop();
            continue;
        }
        visited += 1;
        if visited > max_array_len {
            return Err(ExecError::OutputTooLarge);
        }
        // The child about to be inspected sits at depth `stack.len()`.
        if stack.len() > max_depth {
            return Err(ExecError::Limit {
                what: format!("exported value nests deeper than {max_depth} levels"),
            });
        }
        let object = top.0.clone();
        let key = top.1[top.2].clone();
        let last = stack.len() - 1;
        stack[last].2 += 1;
        if let Some(child) = stored(&object, &key)
            && let Some(child) = child.as_object()
        {
            let child = child.clone();
            if too_long(&child) {
                return Err(too_wide());
            }
            let child_keys = keys(&child, context);
            stack.push((child, child_keys, 0));
        }
    }
}

/// Map a JS error to an execution error, recognising the structural limits.
/// Boa's messages: "Maximum loop iteration limit N exceeded",
/// "exceeded maximum number of recursive calls", "exceeded maximum call stack length".
/// `max` bounds the stored message, so a thrown error can't smuggle output past
/// the cap that bounds `result` and `logs` (see [`cap_message`]).
fn classify(message: String, max: usize) -> ExecError {
    let message = cap_message(message, max);
    let lower = message.to_lowercase();
    let is_limit = lower.contains("loop iteration limit")
        || lower.contains("exceeded maximum number of recursive calls")
        || lower.contains("exceeded maximum call stack length");
    if is_limit {
        ExecError::Limit { what: message }
    } else {
        ExecError::Exception { message }
    }
}

/// Cap a model-facing error message to roughly `max` bytes (cutting on a char
/// boundary). `result` and `logs` are already byte-capped; without this a thrown
/// `Error` with a multi-megabyte `message`, or a giant primitive throw, would
/// flow into the `Outcome` error uncapped and flood the model just the same.
fn cap_message(message: String, max: usize) -> String {
    if message.len() <= max {
        return message;
    }
    let mut end = max;
    while end > 0 && !message.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = message;
    out.truncate(end);
    out.push_str(" [error truncated]");
    out
}

/// Render a promise rejection reason to a string WITHOUT invoking user code.
/// This runs after the wall-clock deadline (the `match promise.state()` in
/// `run_inner` is outside the timeout), so calling `to_string`/`to_primitive` on
/// an object reason would execute an attacker's `toString`/`valueOf`/
/// `Symbol.toPrimitive` unbounded past the timeout, the same post-deadline hazard
/// `result_convertible` guards against on the fulfilled path. For an object we
/// read only its stored `name`/`message` data properties (accessors and Proxy
/// traps yield nothing); for a primitive, `to_string` converts directly and runs
/// no user code.
fn render_rejection(err: &JsValue, context: &mut Context) -> String {
    let Some(obj) = err.as_object() else {
        return err
            .to_string(context)
            .map(|s| s.to_std_string_escaped())
            .unwrap_or_else(|_| "uncaught exception".to_string());
    };
    let data_string = |key: JsString| -> Option<String> {
        obj.borrow()
            .properties()
            .get(&key.into())
            .and_then(|d| d.value().cloned())
            .and_then(|v| v.as_string().map(|s| s.to_std_string_escaped()))
    };
    match (
        data_string(js_string!("name")),
        data_string(js_string!("message")),
    ) {
        (Some(name), Some(message)) => format!("{name}: {message}"),
        (None, Some(message)) => message,
        (Some(name), None) => name,
        (None, None) => "uncaught exception (non-error throw)".to_string(),
    }
}

fn install_globals(context: &mut Context) -> JsResult<()> {
    context.register_global_builtin_callable(
        js_string!("__codemodeCall"),
        3,
        NativeFunction::from_async_fn(bridge_call),
    )?;
    context.register_global_builtin_callable(
        js_string!("__codemodeLog"),
        1,
        NativeFunction::from_fn_ptr(log),
    )?;
    context.eval(Source::from_bytes(
        b"globalThis.console = { log(...a){ __codemodeLog(a.map(String).join(' ')); }, \
           error(...a){ __codemodeLog(a.map(String).join(' ')); }, \
           warn(...a){ __codemodeLog(a.map(String).join(' ')); }, \
           info(...a){ __codemodeLog(a.map(String).join(' ')); } };",
    ))?;
    Ok(())
}

fn log(_this: &JsValue, args: &[JsValue], context: &mut Context) -> JsResult<JsValue> {
    // `to_string` on a JS string is cheap (no copy); converting to a Rust String
    // is the allocation. Check the budget against the JS length first so a giant
    // line (`'x'.repeat(1e9)`) is never copied into Rust just to be discarded.
    let js = args.get_or_undefined(0).to_string(context)?;
    let Some(state) = context.get_data::<RunState>() else {
        return Ok(JsValue::undefined());
    };
    let remaining = state.log_budget.get();
    if remaining == 0 {
        return Ok(JsValue::undefined());
    }
    // `js.len()` is UTF-16 code units, which is a lower bound on the UTF-8 byte
    // length (each code unit is at least one byte), so a giant string is rejected
    // before we ever copy it into Rust. The budget is in bytes, so once we know
    // it's within bounds by that lower bound we account by the exact byte length.
    let truncated = if js.len() > remaining {
        true
    } else {
        let line = js.to_std_string_escaped();
        if line.len() > remaining {
            true
        } else {
            // Charge at least one byte per line so empty/tiny lines still draw
            // down the budget; otherwise `console.log('')` in a loop grows the
            // `logs` Vec without bound. `max(1) <= remaining` here (remaining > 0).
            state.log_budget.set(remaining - line.len().max(1));
            state.logs.borrow_mut().push(line);
            false
        }
    };
    if truncated {
        state.log_budget.set(0);
        state.logs.borrow_mut().push("[logs truncated]".to_string());
    }
    Ok(JsValue::undefined())
}

/// The async native function backing every generated tool call. Reads the
/// per-run bridge from the context's `RunState` and forwards `(server, tool, args)`.
fn bridge_call(
    _this: &JsValue,
    args: &[JsValue],
    context: &RefCell<&mut Context>,
) -> impl std::future::Future<Output = JsResult<JsValue>> {
    let (server, tool, args_json, bridge) = {
        let mut ctx = context.borrow_mut();
        let server = string_arg(args, 0, &mut ctx);
        let tool = string_arg(args, 1, &mut ctx);
        let args_json = string_arg(args, 2, &mut ctx);
        let bridge = ctx.get_data::<RunState>().map(|s| s.bridge.clone());
        (server, tool, args_json, bridge)
    };

    async move {
        let bridge =
            bridge.ok_or_else(|| JsNativeError::typ().with_message("bridge unavailable"))?;
        // Args arrive JSON-stringified (the generated wrappers do this); parse in
        // Rust. A non-JSON argument is a clear error, not a silent null.
        let value: serde_json::Value = serde_json::from_str(&args_json).map_err(|_| {
            JsNativeError::typ().with_message("tool arguments must be a JSON string")
        })?;
        match bridge(server, tool, value).await {
            Ok(result) => {
                let mut ctx = context.borrow_mut();
                JsValue::from_json(&result, &mut ctx)
            }
            Err(err) => Err(JsNativeError::typ().with_message(err.to_string()).into()),
        }
    }
}

fn string_arg(args: &[JsValue], i: usize, context: &mut Context) -> String {
    args.get_or_undefined(i)
        .to_string(context)
        .map(|s| s.to_std_string_escaped())
        .unwrap_or_default()
}

/// Resolves `./servers/<name>` to a generated module; rejects everything else.
struct ServerModuleLoader {
    sources: HashMap<String, String>,
    names: Vec<String>,
}

impl ServerModuleLoader {
    fn new(servers: &[ServerTools]) -> Self {
        let sources = servers
            .iter()
            .map(|s| (format!("./servers/{}", s.name), module_source(s)))
            .collect();
        let names = servers.iter().map(|s| s.name.clone()).collect();
        Self { sources, names }
    }
}

impl ModuleLoader for ServerModuleLoader {
    async fn load_imported_module(
        self: Rc<Self>,
        _referrer: Referrer,
        specifier: JsString,
        context: &RefCell<&mut Context>,
    ) -> JsResult<Module> {
        let spec = specifier.to_std_string_escaped();
        match self.sources.get(&spec) {
            Some(src) => Module::parse(
                Source::from_bytes(src.as_bytes()),
                None,
                &mut context.borrow_mut(),
            ),
            // Name the available servers and show the exact import shape, so a
            // wrong name or a missing `./servers/` prefix self-corrects.
            None => Err(JsNativeError::typ()
                .with_message(format!(
                    "module not found: {spec}. Import a server as \
                     `import * as x from './servers/<name>'`; available servers: {}",
                    available(&self.names),
                ))
                .into()),
        }
    }
}

fn available(names: &[String]) -> String {
    if names.is_empty() {
        "(none)".to_string()
    } else {
        names.join(", ")
    }
}

/// Generate the JS module for one server: one exported function per tool, each
/// forwarding to the bridge. Args are JSON-stringified in JS and parsed in Rust,
/// so prototype pollution can't corrupt what we forward.
fn module_source(server: &ServerTools) -> String {
    let mut out = String::from("const __c = globalThis.__codemodeCall;\n");
    let server_lit = json_lit(&server.name);
    for (i, tool) in server.tools.iter().enumerate() {
        let tool_lit = json_lit(&tool.name);
        out.push_str(&jsdoc(tool));
        // Define under an internal identifier, then export under the exact tool
        // name via a string-named export. This works for any name: a normal name
        // like `echo` is still reached as `ns.echo`, an odd one as `ns["a-b"]`,
        // and sidesteps JS-identifier and reserved-word edge cases entirely.
        out.push_str(&format!(
            "function __t{i}(args) {{ return __c({server_lit}, {tool_lit}, JSON.stringify(args === undefined ? {{}} : args)); }}\n"
        ));
        out.push_str(&format!("export {{ __t{i} as {tool_lit} }};\n"));
    }
    out
}

fn json_lit(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into())
}

/// The JSDoc block for one tool: its description plus, when the schema has
/// properties, a `@param` line describing the single argument object's shape
/// (field names, rough types, optionality). It documents the generated wrapper
/// and gives a typed view for any signature consumer; it never reaches the model
/// as tokens (generated module source isn't sent). Returns "" when there's
/// nothing to say. The whole body is `*/`-sanitized, so a hostile tool name,
/// description, or field name can't break out of the comment.
fn jsdoc(tool: &ToolInfo) -> String {
    let mut body = String::new();
    if let Some(desc) = &tool.description {
        body.push_str(desc);
    }
    if let Some(sig) = arg_signature(&tool.input_schema) {
        if !body.is_empty() {
            body.push('\n');
        }
        body.push_str("@param args ");
        body.push_str(&sig);
    }
    if body.is_empty() {
        return String::new();
    }
    let body = body.replace("*/", "* /");
    let mut out = String::from("/**\n");
    for line in body.lines() {
        out.push_str(" * ");
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(" */\n");
    out
}

/// A compact `{ field: type, optional?: type }` view of a tool's single argument
/// object, derived from the schema's top-level `properties`/`required`. Returns
/// `None` when there are no properties to describe.
fn arg_signature(schema: &serde_json::Value) -> Option<String> {
    let props = schema.get("properties")?.as_object()?;
    if props.is_empty() {
        return None;
    }
    let required: std::collections::HashSet<&str> = schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    let fields: Vec<String> = props
        .iter()
        .map(|(name, spec)| {
            let opt = if required.contains(name.as_str()) {
                ""
            } else {
                "?"
            };
            format!("{name}{opt}: {}", json_type_name(spec))
        })
        .collect();
    Some(format!("{{ {} }}", fields.join(", ")))
}

/// Map a property schema to a rough JS-ish type name for the JSDoc signature.
fn json_type_name(spec: &serde_json::Value) -> &'static str {
    let name = match spec.get("type") {
        Some(serde_json::Value::String(s)) => s.as_str(),
        Some(serde_json::Value::Array(a)) => a.iter().find_map(|v| v.as_str()).unwrap_or("any"),
        _ => return "any",
    };
    match name {
        "string" => "string",
        "number" | "integer" => "number",
        "boolean" => "boolean",
        "array" => "array",
        "object" => "object",
        "null" => "null",
        _ => "any",
    }
}

// --- Event loop: a tokio-driven JobExecutor, adapted from Boa's
// `tokio_event_loop` example (v0.21.1). ---

#[derive(Default)]
struct Queue {
    async_jobs: RefCell<VecDeque<NativeAsyncJob>>,
    promise_jobs: RefCell<VecDeque<PromiseJob>>,
    timeout_jobs: RefCell<BTreeMap<JsInstant, TimeoutJob>>,
    generic_jobs: RefCell<VecDeque<GenericJob>>,
}

impl Queue {
    fn drain_timeout_jobs(&self, context: &mut Context) {
        let now = context.clock().now();
        let mut borrow = self.timeout_jobs.borrow_mut();
        let mut keep = borrow.split_off(&now);
        keep.retain(|_, job| !job.is_cancelled());
        let run = std::mem::replace(borrow.deref_mut(), keep);
        drop(borrow);
        for job in run.into_values() {
            if let Err(e) = job.call(context) {
                eprintln!("Uncaught {e}");
            }
        }
    }

    fn drain_jobs(&self, context: &mut Context) {
        self.drain_timeout_jobs(context);
        // Pull the job out and drop the borrow before calling it: in edition 2024
        // a let-chain scrutinee's temporary lives to the end of the `if`, so
        // borrowing inline would hold the `RefMut` across `call`, which can
        // re-enqueue and re-borrow `generic_jobs`.
        let generic = self.generic_jobs.borrow_mut().pop_front();
        if let Some(generic) = generic
            && let Err(err) = generic.call(context)
        {
            eprintln!("Uncaught {err}");
        }
        let jobs = std::mem::take(&mut *self.promise_jobs.borrow_mut());
        for job in jobs {
            if let Err(e) = job.call(context) {
                eprintln!("Uncaught {e}");
            }
        }
        context.clear_kept_objects();
    }
}

impl JobExecutor for Queue {
    fn enqueue_job(self: Rc<Self>, job: Job, context: &mut Context) {
        match job {
            Job::PromiseJob(job) => self.promise_jobs.borrow_mut().push_back(job),
            Job::AsyncJob(job) => self.async_jobs.borrow_mut().push_back(job),
            Job::TimeoutJob(t) => {
                let now = context.clock().now();
                self.timeout_jobs.borrow_mut().insert(now + t.timeout(), t);
            }
            Job::GenericJob(g) => self.generic_jobs.borrow_mut().push_back(g),
            _ => {}
        }
    }

    fn run_jobs(self: Rc<Self>, context: &mut Context) -> JsResult<()> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| JsNativeError::typ().with_message(e.to_string()))?;
        tokio::task::LocalSet::default()
            .block_on(&runtime, self.run_jobs_async(&RefCell::new(context)))
    }

    async fn run_jobs_async(self: Rc<Self>, context: &RefCell<&mut Context>) -> JsResult<()> {
        use futures_concurrency::future::FutureGroup;
        use futures_lite::StreamExt;

        let mut group = FutureGroup::new();
        loop {
            // Run every ready microtask / timeout / generic job, then pick up any
            // async jobs they enqueued.
            self.drain_jobs(&mut context.borrow_mut());
            for job in std::mem::take(&mut *self.async_jobs.borrow_mut()) {
                group.insert(job.call(context));
            }

            // Ready synchronous work (promise/generic jobs) should drain on the
            // next iteration. A timeout job that isn't due yet is *not* ready: we
            // must wait for its instant, not spin.
            let has_ready_micro =
                !self.promise_jobs.borrow().is_empty() || !self.generic_jobs.borrow().is_empty();
            let next_timeout_ms = self
                .timeout_jobs
                .borrow()
                .keys()
                .next()
                .map(|i| i.millis_since_epoch());
            let has_async = !group.is_empty();

            if !has_ready_micro && !has_async && next_timeout_ms.is_none() {
                return Ok(());
            }
            if has_ready_micro {
                // More synchronous work queued; loop to drain it.
                tokio::task::yield_now().await;
                continue;
            }
            // Only async jobs and/or a future timer remain. Sleep until the
            // nearest timer is due, or until an async job completes, whichever
            // is first, instead of busy-polling, so a pending `setTimeout` can't
            // peg the island's CPU (and starve other executions sharing it).
            let sleep_ms = next_timeout_ms.map(|due| {
                let now = context.borrow().clock().now().millis_since_epoch();
                due.saturating_sub(now)
            });
            match (has_async, sleep_ms) {
                (true, Some(ms)) => {
                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_millis(ms)) => {}
                        res = group.next() => {
                            if let Some(Err(err)) = res {
                                eprintln!("Uncaught {err}");
                            }
                        }
                    }
                }
                (true, None) => {
                    if let Some(Err(err)) = group.next().await {
                        eprintln!("Uncaught {err}");
                    }
                }
                (false, Some(ms)) => tokio::time::sleep(Duration::from_millis(ms)).await,
                (false, None) => return Ok(()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use serde_json::json;

    use super::*;
    use crate::runtime::BridgeError;
    use crate::types::ToolInfo;

    fn demo_servers() -> Vec<ServerTools> {
        vec![ServerTools {
            name: "demo".to_string(),
            tools: vec![ToolInfo {
                name: "echo".to_string(),
                description: None,
                input_schema: json!({ "type": "object" }),
            }],
        }]
    }

    fn run(source: &str, servers: Vec<ServerTools>, bridge: Bridge) -> Outcome {
        run_with(source, servers, bridge, Limits::default())
    }

    fn run_with(
        source: &str,
        servers: Vec<ServerTools>,
        bridge: Bridge,
        limits: Limits,
    ) -> Outcome {
        let request = RunRequest {
            source: source.to_string(),
            servers,
            bridge,
            limits,
        };
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        tokio::task::LocalSet::new().block_on(&rt, Boa::new().run(request))
    }

    fn echo_bridge() -> Bridge {
        Rc::new(|_s, _t, args| Box::pin(async move { Ok(args) }))
    }

    // The conformance suite (the cross-backend contract) runs as an integration
    // test in `tests/conformance.rs`; the unit tests here cover Boa's own
    // mechanics directly without depending on it.

    #[test]
    fn imports_and_calls_a_tool() {
        let bridge: Bridge = Rc::new(|_server, _tool, args| Box::pin(async move { Ok(args) }));
        let outcome = run(
            "import * as demo from './servers/demo';\n\
             const r = await demo.echo({ hi: 1 });\n\
             export default r;",
            demo_servers(),
            bridge,
        );
        assert!(
            outcome.error.is_none(),
            "unexpected error: {:?}",
            outcome.error
        );
        assert_eq!(outcome.result, json!({ "hi": 1 }));
    }

    #[test]
    fn captures_console_log() {
        let bridge: Bridge = Rc::new(|_s, _t, args| Box::pin(async move { Ok(args) }));
        let outcome = run(
            "console.log('hello', 42); export default 1;",
            vec![],
            bridge,
        );
        assert!(outcome.error.is_none());
        assert_eq!(outcome.logs, vec!["hello 42".to_string()]);
    }

    #[test]
    fn handles_non_identifier_tool_names() {
        let bridge: Bridge = Rc::new(|_s, tool, _a| {
            Box::pin(async move { Ok(serde_json::json!({ "called": tool })) })
        });
        let servers = vec![ServerTools {
            name: "demo".to_string(),
            tools: vec![ToolInfo {
                name: "weird-tool.name".to_string(),
                description: None,
                input_schema: json!({}),
            }],
        }];
        let outcome = run(
            "import * as demo from './servers/demo';\n\
             const r = await demo['weird-tool.name']({});\n\
             export default r;",
            servers,
            bridge,
        );
        assert!(outcome.error.is_none(), "error: {:?}", outcome.error);
        assert_eq!(outcome.result, json!({ "called": "weird-tool.name" }));
    }

    #[test]
    fn module_source_emits_typed_param_signature() {
        let server = ServerTools {
            name: "fs".to_string(),
            tools: vec![ToolInfo {
                name: "readFile".to_string(),
                description: Some("Read a file".to_string()),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "encoding": { "type": "string" },
                        "lines": { "type": "integer" }
                    },
                    "required": ["path"]
                }),
            }],
        };
        let src = module_source(&server);
        assert!(src.contains("Read a file"), "missing description: {src}");
        assert!(
            src.contains("@param args { path: string, encoding?: string, lines?: number }"),
            "missing/incorrect signature: {src}"
        );
    }

    #[test]
    fn module_source_sanitizes_comment_breakout_in_field_names() {
        // A hostile MCP server could name a field to close the JSDoc comment.
        let server = ServerTools {
            name: "x".to_string(),
            tools: vec![ToolInfo {
                name: "t".to_string(),
                description: None,
                input_schema: json!({ "type": "object", "properties": { "*/ evil": { "type": "string" } } }),
            }],
        };
        let src = module_source(&server);
        // The only `*/` left is the comment terminator, never the injected name.
        assert!(
            !src.contains("*/ evil"),
            "comment breakout not neutralized: {src}"
        );
    }

    #[test]
    fn unknown_server_import_fails() {
        let bridge: Bridge = Rc::new(|_s, _t, args| Box::pin(async move { Ok(args) }));
        let outcome = run(
            "import * as x from './servers/nope'; export default 1;",
            demo_servers(),
            bridge,
        );
        let Some(ExecError::Exception { message }) = outcome.error else {
            panic!("expected a Js error, got {:?}", outcome.error);
        };
        // The message should guide the model back on track: name the available
        // servers so it can pick a real one rather than guess again.
        assert!(
            message.contains("demo"),
            "should list available servers: {message}"
        );
    }

    #[test]
    fn missing_default_export_is_a_clear_error() {
        // The most common LLM mistake: compute the answer but never `export
        // default` it. Returning a silent `null` gives the model nothing to fix,
        // so it retries blindly. A clear error names the fix.
        let outcome = run(
            "const answer = 41 + 1; console.log(answer);",
            vec![],
            echo_bridge(),
        );
        let Some(ExecError::Exception { message }) = outcome.error else {
            panic!("expected a Js error, got {:?}", outcome.error);
        };
        assert!(
            message.contains("export default"),
            "should name the fix: {message}"
        );
    }

    #[test]
    fn explicit_null_export_is_allowed() {
        // `export default null` is a real answer, not the missing-export mistake.
        let outcome = run("export default null;", vec![], echo_bridge());
        assert!(
            outcome.error.is_none(),
            "unexpected error: {:?}",
            outcome.error
        );
        assert_eq!(outcome.result, json!(null));
    }

    #[test]
    fn sandbox_exposes_no_host_globals() {
        // None of the usual escape hatches should exist: the engine is sandboxed
        // by construction and we register only the MCP door and a console shim.
        let outcome = run(
            "const present = [];\n\
             for (const n of ['fetch','process','require','Deno','XMLHttpRequest','WebSocket','global']) {\n\
               if (typeof globalThis[n] !== 'undefined') present.push(n);\n\
             }\n\
             export default present;",
            vec![],
            echo_bridge(),
        );
        assert!(outcome.error.is_none(), "{:?}", outcome.error);
        assert_eq!(outcome.result, json!([]));
    }

    #[test]
    fn bridge_global_is_reachable_but_not_enumerable() {
        // The bridge must stay reachable (the generated modules call it) yet not
        // show up in `Object.keys(globalThis)`, so model code enumerating the
        // globals can't discover the door. Pins a documented invariant.
        let outcome = run(
            "export default {\n\
               reachable: typeof __codemodeCall === 'function',\n\
               enumerable: Object.keys(globalThis).includes('__codemodeCall'),\n\
             };",
            vec![],
            echo_bridge(),
        );
        assert!(outcome.error.is_none(), "{:?}", outcome.error);
        assert_eq!(
            outcome.result,
            json!({ "reachable": true, "enumerable": false })
        );
    }

    #[test]
    fn frees_memory_between_runs() {
        // A run that allocates a large array completes, and the next run is
        // unaffected: the per-run context (and its heap) was dropped.
        let big = run(
            "const a = new Array(500000).fill(7); export default a.length;",
            vec![],
            echo_bridge(),
        );
        assert!(big.error.is_none(), "{:?}", big.error);
        assert_eq!(big.result, json!(500000));

        let after = run("export default 1 + 1;", vec![], echo_bridge());
        assert!(after.error.is_none());
        assert_eq!(after.result, json!(2));
    }

    #[test]
    fn infinite_loop_is_caught_not_panicking() {
        // A runtime-limit hit during module evaluation makes Boa panic internally;
        // we must catch it and return a clean Limit error, keeping the island alive.
        let limits = Limits {
            max_loop_iterations: 100_000,
            ..Limits::default()
        };
        let outcome = run_with(
            "while (true) {} export default 1;",
            vec![],
            echo_bridge(),
            limits,
        );
        assert!(
            matches!(outcome.error, Some(ExecError::Limit { .. })),
            "expected a Limit error, got {:?}",
            outcome.error
        );
        // And the engine still works for a subsequent run (island not dead).
        let after = run("export default 2 + 2;", vec![], echo_bridge());
        assert_eq!(after.result, json!(4));
    }

    #[test]
    fn rejects_oversized_source() {
        let src = "a".repeat(MAX_SOURCE_BYTES + 1);
        let outcome = run(&src, vec![], echo_bridge());
        assert!(
            matches!(outcome.error, Some(ExecError::Limit { .. })),
            "expected a Limit error, got {:?}",
            outcome.error
        );
    }

    #[test]
    fn recursion_limit_trips() {
        let limits = Limits {
            max_recursion_depth: 20,
            ..Limits::default()
        };
        let outcome = run_with(
            "function r(n){ return n > 0 ? r(n - 1) + 1 : 0; } export default r(100);",
            vec![],
            echo_bridge(),
            limits,
        );
        assert!(
            matches!(outcome.error, Some(ExecError::Limit { .. })),
            "expected a Limit error, got {:?}",
            outcome.error
        );
    }

    #[test]
    fn rejects_oversized_result() {
        let limits = Limits {
            max_output_bytes: 256,
            ..Limits::default()
        };
        let outcome = run_with(
            "export default 'x'.repeat(10000);",
            vec![],
            echo_bridge(),
            limits,
        );
        assert!(matches!(outcome.error, Some(ExecError::OutputTooLarge)));
    }

    #[test]
    fn caps_oversized_exception_message() {
        // `result` and `logs` are byte-capped; a thrown error's message must be
        // too, or untrusted code floods the model with `throw new Error(huge)`.
        let limits = Limits {
            max_output_bytes: 256,
            ..Limits::default()
        };
        let outcome = run_with(
            "throw new Error('x'.repeat(1000000));",
            vec![],
            echo_bridge(),
            limits,
        );
        let Some(ExecError::Exception { message }) = outcome.error else {
            panic!("expected an exception, got {:?}", outcome.error);
        };
        assert!(
            message.len() < 512,
            "exception message not capped: {} bytes",
            message.len()
        );
    }

    #[test]
    fn caps_oversized_primitive_throw() {
        // The same cap must apply to a non-Error throw (a giant primitive string),
        // which renders through `to_string` rather than the stored `message`.
        let limits = Limits {
            max_output_bytes: 256,
            ..Limits::default()
        };
        let outcome = run_with("throw 'x'.repeat(1000000);", vec![], echo_bridge(), limits);
        let Some(ExecError::Exception { message }) = outcome.error else {
            panic!("expected an exception, got {:?}", outcome.error);
        };
        assert!(
            message.len() < 512,
            "primitive throw message not capped: {} bytes",
            message.len()
        );
    }

    #[test]
    fn caps_log_output() {
        let limits = Limits {
            max_output_bytes: 200,
            ..Limits::default()
        };
        let outcome = run_with(
            "for (let i = 0; i < 1000; i++) console.log('noisy log line number ' + i);\n\
             export default 'done';",
            vec![],
            echo_bridge(),
            limits,
        );
        assert!(outcome.error.is_none(), "{:?}", outcome.error);
        let bytes: usize = outcome.logs.iter().map(|l| l.len()).sum();
        assert!(bytes <= 256, "logs not capped: {bytes} bytes");
        assert_eq!(
            outcome.logs.last().map(String::as_str),
            Some("[logs truncated]")
        );
    }

    #[test]
    fn empty_log_lines_are_budgeted() {
        // Empty lines must still draw down the budget, or `console.log('')` in a
        // loop grows the logs Vec without bound (would be a memory DoS).
        let limits = Limits {
            max_output_bytes: 10,
            ..Limits::default()
        };
        let outcome = run_with(
            "for (let i = 0; i < 1000; i++) console.log(''); export default 'done';",
            vec![],
            echo_bridge(),
            limits,
        );
        assert!(outcome.error.is_none(), "{:?}", outcome.error);
        assert!(
            outcome.logs.len() <= 11,
            "empty logs not bounded: {} lines",
            outcome.logs.len()
        );
    }

    #[test]
    fn rejects_pathologically_deep_results() {
        // A deeply nested export would overflow the native stack during `to_json`
        // and *abort* the process; the depth guard turns it into a clean Limit.
        // (1000 is past the 256 cap but shallow enough that the structure's own
        // drop is safe.)
        let outcome = run(
            "let a = []; for (let i = 0; i < 1000; i++) a = [a]; export default a;",
            vec![],
            echo_bridge(),
        );
        assert!(
            matches!(outcome.error, Some(ExecError::Limit { .. })),
            "expected a Limit error, got {:?}",
            outcome.error
        );
    }

    #[test]
    fn rejects_deep_results_hidden_behind_wide_padding() {
        // Many shallow keys before a deep chain must NOT let the chain slip past
        // the depth probe; a width-budgeted probe would early-accept here and
        // then abort the process inside to_json.
        let outcome = run(
            "const root = {};\n\
             for (let i = 0; i < 120000; i++) root['p' + i] = 1;\n\
             let chain = []; for (let i = 0; i < 1000; i++) chain = [chain];\n\
             root.z_deep = chain;\n\
             export default root;",
            vec![],
            echo_bridge(),
        );
        assert!(
            matches!(outcome.error, Some(ExecError::Limit { .. })),
            "expected a Limit error, got {:?}",
            outcome.error
        );
    }

    #[test]
    fn rejects_dag_shaped_results_without_hanging() {
        // A shallow DAG of shared sub-objects: only ~60 objects exist, but every
        // level points at the same child twice, so the graph has 2^60 distinct
        // root-to-leaf paths. `to_json` re-expands a shared node once per
        // reference, so without a work cap both the convertibility probe and
        // `to_json` would walk exponentially many paths after the deadline and
        // hang the island. The visit cap must turn this into a clean, fast error.
        let started = std::time::Instant::now();
        let outcome = run(
            "let x = { a: 1 };\n\
             for (let i = 0; i < 60; i++) x = { l: x, r: x };\n\
             export default x;",
            vec![],
            echo_bridge(),
        );
        assert!(
            matches!(outcome.error, Some(ExecError::OutputTooLarge)),
            "expected OutputTooLarge, got {:?}",
            outcome.error
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "DAG rejection should be fast, took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn accepts_benign_shared_subobjects() {
        // Plain (non-exponential) sharing is legitimate and must still convert:
        // `to_json` serializes the shared object in each place, well within the
        // output cap, so the visit cap must not reject it.
        let outcome = run(
            "const shared = { x: 1 };\n\
             export default { a: shared, b: shared, c: shared };",
            vec![],
            echo_bridge(),
        );
        assert!(
            outcome.error.is_none(),
            "unexpected error: {:?}",
            outcome.error
        );
        assert_eq!(
            outcome.result,
            json!({ "a": { "x": 1 }, "b": { "x": 1 }, "c": { "x": 1 } })
        );
    }

    #[test]
    fn rejects_huge_sparse_array_results() {
        // A sparse array with a giant `.length` would make `to_json` eagerly
        // `Vec::with_capacity` that length (tens of GB) and abort the process; it
        // must come back as a clean Limit instead.
        let outcome = run(
            "const a = []; a.length = 4294967295; export default a;",
            vec![],
            echo_bridge(),
        );
        assert!(
            matches!(outcome.error, Some(ExecError::Limit { .. })),
            "expected a Limit error, got {:?}",
            outcome.error
        );
    }

    #[test]
    fn allows_normally_nested_results() {
        // Ordinary nesting (well under the cap) round-trips fine.
        let outcome = run(
            "export default { a: { b: { c: [1, 2, { d: 3 }] } } };",
            vec![],
            echo_bridge(),
        );
        assert!(outcome.error.is_none(), "{:?}", outcome.error);
        assert_eq!(
            outcome.result,
            json!({ "a": { "b": { "c": [1, 2, { "d": 3 }] } } })
        );
    }

    #[test]
    fn getters_are_not_invoked_during_result_conversion() {
        // The result guard runs after the wall-clock deadline. If it read values
        // with `get` it would fire getters: arbitrary code past the timeout, a
        // DoS. It must read stored values only, matching `to_json` (which renders
        // an accessor as null and never calls it). Here a getter, if invoked,
        // would replace `payload` with a 400-deep object (past the 256 cap) and
        // trip a Limit. `trigger` is defined first so it would be visited before
        // `payload`. The run must instead succeed with the accessor as null.
        let outcome = run(
            "function deep(n) { let o = {}; for (let i = 0; i < n; i++) o = { next: o }; return o; }\n\
             const o = {};\n\
             Object.defineProperty(o, 'trigger', { enumerable: true, get() { o.payload = deep(400); return 0; } });\n\
             o.payload = 1;\n\
             export default o;",
            vec![],
            echo_bridge(),
        );
        assert!(
            outcome.error.is_none(),
            "getter must not run during conversion, got {:?}",
            outcome.error
        );
        assert_eq!(outcome.result, json!({ "trigger": null, "payload": 1 }));
    }

    #[test]
    fn rejection_reason_tostring_is_not_invoked() {
        // The rejection-rendering arm runs after the wall-clock deadline, so a
        // thrown object's `toString` must NOT run there (it would execute user JS
        // unbounded past the timeout, the same hazard the fulfilled path guards).
        // We render from the stored `name`/`message` data properties instead. If
        // `toString` ran, the message would be "TOSTRING-RAN".
        let outcome = run(
            "throw { name: 'BadThing', message: 'boom', \
             toString() { return 'TOSTRING-RAN'; } };\n\
             export default 1;",
            vec![],
            echo_bridge(),
        );
        match outcome.error {
            Some(ExecError::Exception { message }) => {
                assert!(
                    message.contains("boom") && message.contains("BadThing"),
                    "expected the stored name/message, got: {message}"
                );
                assert!(
                    !message.contains("TOSTRING-RAN"),
                    "toString was invoked during rejection rendering: {message}"
                );
            }
            other => panic!("expected an Exception, got {other:?}"),
        }
    }

    #[test]
    fn rejection_with_primitive_reason_renders() {
        // A thrown primitive (no object, no user code) still renders sensibly.
        let outcome = run("throw 'plain string error';", vec![], echo_bridge());
        assert!(
            matches!(&outcome.error, Some(ExecError::Exception { message }) if message.contains("plain string error")),
            "got: {:?}",
            outcome.error
        );
    }

    #[test]
    fn bridge_error_surfaces() {
        let bridge: Bridge =
            Rc::new(|_s, _t, _a| Box::pin(async move { Err(BridgeError::Call("boom".into())) }));
        let outcome = run(
            "import * as demo from './servers/demo';\n\
             await demo.echo({}); export default 'unreached';",
            demo_servers(),
            bridge,
        );
        assert!(matches!(outcome.error, Some(ExecError::Exception { .. })));
    }
}
