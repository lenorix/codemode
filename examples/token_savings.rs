//! Token usage and latency: traditional tool-calling vs. code mode.
//!
//! A small but production-shaped task: a "restock-priority" report over a
//! catalog, where answering means chaining five tools per product (list, then
//! reviews, inventory, pricing, sales) and filtering and ranking the result.
//! This is the kind of multi-tool pipeline a real agent runs, and it is where
//! the two strategies diverge sharply.
//!
//! Runs the *same* task two ways against a local OpenAI-compatible LLM and
//! reports, for each, the total tokens, the number of LLM turns, and the
//! wall-clock time split into LLM time and tool time:
//!
//!   1. Traditional: the model calls tools one by one; every intermediate
//!      result is fed back into its context and re-tokenized on each turn.
//!   2. Code mode: the model writes one program (via the `execute` tool) that
//!      chains the calls itself; only the final result returns to the model.
//!
//! Each LLM turn is a network round-trip, so fewer turns means lower latency.
//! Code mode collapses the orchestration into a single turn, so it should win
//! on both tokens and time even though the tool work itself is identical.
//!
//! Start an OpenAI-compatible server (e.g. LM Studio) at 127.0.0.1:1234, then:
//!
//!   cargo run --example token_savings
//!
//! The tools and their data are identical across both runs — the only thing
//! that changes is whether the model orchestrates them by calling or by coding.

use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use codemode_mcp::{CodeMode, LocalTools};
use serde_json::{Value, json};

const ENDPOINT: &str = "http://127.0.0.1:1234/v1/chat/completions";
const MODEL: &str = "qwen/qwen3.5-9b";
const MAX_TURNS: usize = 25;

// --- The shared dataset and tool handlers (used by BOTH runs) ---

/// The product catalog, loaded from `catalog.json` next to this example. Each
/// record holds everything about a product; the five tools below each expose a
/// different slice of it, the way separate MCP servers would. Keeping the data
/// in JSON keeps this file about the comparison, not the fixtures, and lets
/// other examples reuse the same dataset.
static CATALOG: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("catalog.json")).expect("catalog.json is valid JSON")
});

fn products() -> &'static [Value] {
    CATALOG["products"]
        .as_array()
        .expect("catalog has a products array")
}

fn product(product_id: &str) -> Option<&'static Value> {
    products().iter().find(|p| p["id"] == product_id)
}

fn list_products() -> Value {
    // Only the fields a listing exposes; the rest is reached through the other tools.
    Value::Array(
        products()
            .iter()
            .map(|p| json!({ "id": p["id"], "name": p["name"], "category": p["category"] }))
            .collect(),
    )
}

fn get_reviews(product_id: &str) -> Value {
    // This data re-enters the model's context on every turn of the traditional
    // run, but never does in code mode. Across products and the five tools, the
    // accumulated results are what dominate the traditional run's token bill.
    product(product_id)
        .map(|p| p["reviews"].clone())
        .unwrap_or_else(|| json!([]))
}

fn get_inventory(product_id: &str) -> Value {
    match product(product_id) {
        Some(p) => json!({ "in_stock": p["in_stock"], "restock_days": p["restock_days"] }),
        None => json!({ "error": "unknown product" }),
    }
}

fn get_pricing(product_id: &str) -> Value {
    match product(product_id) {
        Some(p) => json!({ "price_cents": p["price_cents"], "currency": "USD" }),
        None => json!({ "error": "unknown product" }),
    }
}

fn get_sales(product_id: &str) -> Value {
    match product(product_id) {
        Some(p) => json!({ "units_sold_30d": p["units_sold_30d"] }),
        None => json!({ "error": "unknown product" }),
    }
}

const TASK: &str = "Produce a top-sellers report. Among products whose average review rating is at least \
    4.5 and that are currently in stock, list the top three by 30-day units sold, best first, each \
    formatted as \"<name> — <units> units at $<price>\" (price in whole dollars).";

// --- Minimal OpenAI chat client ---

async fn chat(
    client: &reqwest::Client,
    messages: &Value,
    tools: &Value,
) -> Result<(Value, u64), String> {
    let body = json!({
        // Thinking models (qwen, gpt-oss) emit a long reasoning trace before the
        // visible answer; too small a cap truncates mid-reasoning and the turn
        // comes back with empty content. Give them room to finish and answer.
        "model": MODEL, "messages": messages, "tools": tools,
        "temperature": 0, "max_tokens": 8000
    });
    let resp = client
        .post(ENDPOINT)
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            let kind = if e.is_timeout() {
                " (timed out)"
            } else if e.is_connect() {
                " (could not connect — is the LLM server running?)"
            } else {
                ""
            };
            format!("request to {ENDPOINT} failed{kind}: {e}")
        })?;
    let v: Value = resp.json().await.map_err(|e| e.to_string())?;
    let tokens = v["usage"]["total_tokens"].as_u64().unwrap_or(0);
    let message = v["choices"][0]["message"].clone();
    if message.is_null() {
        return Err(format!("unexpected response: {v}"));
    }
    Ok((message, tokens))
}

/// The assistant turn to replay into history: `content`, any `tool_calls`, and
/// crucially the model's reasoning field (`reasoning` / `reasoning_content`).
/// Thinking models such as qwen put their plan and partial results in reasoning
/// and leave `content` empty while they work through tool calls. Dropping it
/// erases their memory of progress, so they re-issue the same calls every turn
/// and never reach an answer; replaying it keeps the multi-step plan coherent.
///
/// This is not a crutch for the comparison: it is what a real agent framework
/// does. Rig preserves the reasoning block in the assistant history it resends
/// (rig-core agent/prompt_request pushes the full assistant turn; the deepseek
/// and moonshot providers read and re-serialize `reasoning_content`, with a test
/// asserting it). So this mirrors normal agent behaviour rather than inventing it.
fn history_entry(message: &Value) -> Value {
    let mut entry = json!({ "role": "assistant", "content": message["content"].clone() });
    for field in ["reasoning", "reasoning_content"] {
        if let Some(value) = message.get(field).filter(|v| !v.is_null()) {
            entry[field] = value.clone();
        }
    }
    if let Some(tool_calls) = message.get("tool_calls").filter(|v| !v.is_null()) {
        entry["tool_calls"] = tool_calls.clone();
    }
    entry
}

/// What a run cost: the answer, total tokens, LLM round-trips ("turns"), and
/// where the wall-clock time went (waiting on the model vs. running tools).
struct RunMetrics {
    answer: String,
    tokens: u64,
    turns: usize,
    llm_time: Duration,
    tool_time: Duration,
}

/// Agent loop: keep going until the model answers with plain text. `dispatch`
/// runs a requested tool call and returns its JSON result.
async fn run_agent<F, Fut>(
    client: &reqwest::Client,
    system: &str,
    tools: Value,
    mut dispatch: F,
) -> Result<RunMetrics, String>
where
    F: FnMut(String, Value) -> Fut,
    Fut: std::future::Future<Output = Value>,
{
    let mut messages = json!([
        { "role": "system", "content": system },
        { "role": "user", "content": TASK }
    ]);
    let mut tokens = 0u64;
    let mut llm_time = Duration::ZERO;
    let mut tool_time = Duration::ZERO;

    for turns in 1..=MAX_TURNS {
        let started = Instant::now();
        let (message, turn_tokens) = chat(client, &messages, &tools).await?;
        llm_time += started.elapsed();
        tokens += turn_tokens;
        messages
            .as_array_mut()
            .unwrap()
            .push(history_entry(&message));

        let tool_calls = message["tool_calls"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if tool_calls.is_empty() {
            let answer = message["content"]
                .as_str()
                .unwrap_or_default()
                .trim()
                .to_string();
            return Ok(RunMetrics {
                answer,
                tokens,
                turns,
                llm_time,
                tool_time,
            });
        }

        for call in tool_calls {
            let name = call["function"]["name"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            let args: Value =
                serde_json::from_str(call["function"]["arguments"].as_str().unwrap_or("{}"))
                    .unwrap_or(json!({}));
            let started = Instant::now();
            let result = dispatch(name, args).await;
            tool_time += started.elapsed();
            messages.as_array_mut().unwrap().push(json!({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result.to_string()
            }));
        }
    }
    Err(format!(
        "did not finish within {MAX_TURNS} turns (tokens so far: {tokens})"
    ))
}

async fn traditional(client: &reqwest::Client) -> Result<RunMetrics, String> {
    // One tool per product attribute, the shape a real MCP server exposes.
    let by_product = |desc: &str| {
        json!({
            "type": "object",
            "properties": { "product_id": { "type": "string" } },
            "required": ["product_id"],
            "description": desc,
        })
    };
    let tools = json!([
        { "type": "function", "function": {
            "name": "list_products",
            "description": "List all products (id, name, category).",
            "parameters": { "type": "object", "properties": {} }
        }},
        { "type": "function", "function": {
            "name": "get_reviews",
            "description": "Get the reviews (rating, text) for one product.",
            "parameters": by_product("the product to fetch reviews for")
        }},
        { "type": "function", "function": {
            "name": "get_inventory",
            "description": "Get stock status (in_stock, restock_days) for one product.",
            "parameters": by_product("the product to check stock for")
        }},
        { "type": "function", "function": {
            "name": "get_pricing",
            "description": "Get pricing (price_cents, currency) for one product.",
            "parameters": by_product("the product to price")
        }},
        { "type": "function", "function": {
            "name": "get_sales",
            "description": "Get recent sales (units_sold_30d) for one product.",
            "parameters": by_product("the product to fetch sales for")
        }}
    ]);
    let system = "You are a data assistant. To answer, you need for each product: its reviews \
                  (for the average rating), its inventory (in stock or not), its pricing (price), and \
                  its 30-day sales (units sold). Use the tools to gather that data, then compute the \
                  result and reply with only the final answer.";
    run_agent(client, system, tools, |name, args| async move {
        let id = args["product_id"].as_str().unwrap_or("");
        match name.as_str() {
            "list_products" => list_products(),
            "get_reviews" => get_reviews(id),
            "get_inventory" => get_inventory(id),
            "get_pricing" => get_pricing(id),
            "get_sales" => get_sales(id),
            other => json!({ "error": format!("unknown tool {other}") }),
        }
    })
    .await
}

async fn code_mode(client: &reqwest::Client, cm: Arc<CodeMode>) -> Result<RunMetrics, String> {
    let tools = json!([
        { "type": "function", "function": {
            "name": "execute",
            "description": "Run a JavaScript ES module and return its `export default` value.",
            "parameters": {
                "type": "object",
                "properties": { "code": { "type": "string" } },
                "required": ["code"]
            }
        }}
    ]);
    let system = "You answer by writing ONE JavaScript program and running it with the `execute` tool. \
                  Available module:\n\
                  import * as store from './servers/store';\n\
                  await store.listProducts() -> [{ id, name, category }]\n\
                  await store.getReviews({ product_id }) -> [{ rating, text }]\n\
                  await store.getInventory({ product_id }) -> { in_stock, restock_days }\n\
                  await store.getPricing({ product_id }) -> { price_cents, currency }\n\
                  await store.getSales({ product_id }) -> { units_sold_30d }\n\
                  Compute the answer in code and `export default` it (top-level await is allowed). \
                  The data-gathering usually looks like this; you finish it to answer the task:\n\
                  ```\n\
                  import * as store from './servers/store';\n\
                  const products = await store.listProducts();\n\
                  const rows = [];\n\
                  for (const p of products) {\n\
                    const [reviews, inv, price, sales] = await Promise.all([\n\
                      store.getReviews({ product_id: p.id }),\n\
                      store.getInventory({ product_id: p.id }),\n\
                      store.getPricing({ product_id: p.id }),\n\
                      store.getSales({ product_id: p.id }),\n\
                    ]);\n\
                    const avg = reviews.reduce((s, r) => s + r.rating, 0) / reviews.length;\n\
                    rows.push({ name: p.name, avg, inv, price, sales });\n\
                  }\n\
                  // ...now filter and rank `rows` as the task asks, then export the answer.\n\
                  export default rows;\n\
                  ```\n\
                  The execute tool returns { result, logs, error }. Call execute ONCE, then reply with only the final answer.";

    run_agent(client, system, tools, move |name, args| {
        let cm = cm.clone();
        async move {
            if name != "execute" {
                return json!({ "error": "unknown tool" });
            }
            let code = args["code"].as_str().unwrap_or("").to_string();
            match cm.execute(&code).await {
                Ok(outcome) => serde_json::to_value(outcome).unwrap_or(Value::Null),
                Err(e) => json!({ "error": e.to_string() }),
            }
        }
    })
    .await
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .unwrap();

    let cm = Arc::new(
        CodeMode::builder()
            .local_tools({
                // The same five tools the traditional run gets, behind one server.
                let by_product = json!({ "type": "object", "properties": { "product_id": { "type": "string" } } });
                LocalTools::new("store")
                    .tool("listProducts", "List all products.", json!({ "type": "object" }), |_args| async move {
                        Ok(list_products())
                    })
                    .tool("getReviews", "Get reviews for a product.", by_product.clone(), |args| async move {
                        Ok(get_reviews(args["product_id"].as_str().unwrap_or("")))
                    })
                    .tool("getInventory", "Get stock status for a product.", by_product.clone(), |args| async move {
                        Ok(get_inventory(args["product_id"].as_str().unwrap_or("")))
                    })
                    .tool("getPricing", "Get pricing for a product.", by_product.clone(), |args| async move {
                        Ok(get_pricing(args["product_id"].as_str().unwrap_or("")))
                    })
                    .tool("getSales", "Get recent sales for a product.", by_product, |args| async move {
                        Ok(get_sales(args["product_id"].as_str().unwrap_or("")))
                    })
            })
            .build()
            .await
            .expect("build code mode"),
    );

    println!("Task: {TASK}\n");

    let traditional = timed("Traditional tool-calling", traditional(&client)).await;
    let code_mode = timed("Code mode", code_mode(&client, cm.clone())).await;

    if let (Some(t), Some(c)) = (&traditional, &code_mode) {
        compare(t, c);
    }
}

/// Run one approach, time its total wall-clock, print its metrics, and return
/// `(RunMetrics, total elapsed)`.
async fn timed(
    label: &str,
    run: impl std::future::Future<Output = Result<RunMetrics, String>>,
) -> Option<(RunMetrics, Duration)> {
    let started = Instant::now();
    let result = run.await;
    let elapsed = started.elapsed();
    println!("== {label} ==");
    match result {
        Ok(m) => {
            println!("answer:      {}", m.answer);
            println!("tokens:      {}", m.tokens);
            println!("LLM turns:   {}", m.turns);
            println!(
                "wall-clock:  {:.1}s   (LLM {:.1}s + tools {:.1}s)\n",
                elapsed.as_secs_f64(),
                m.llm_time.as_secs_f64(),
                m.tool_time.as_secs_f64(),
            );
            Some((m, elapsed))
        }
        Err(e) => {
            println!("error: {e}\n");
            None
        }
    }
}

fn compare((trad, trad_time): &(RunMetrics, Duration), (cm, cm_time): &(RunMetrics, Duration)) {
    let pct = |base: f64, now: f64| {
        if base > 0.0 {
            (base - now) / base * 100.0
        } else {
            0.0
        }
    };
    println!("== Result (traditional -> code mode) ==");
    println!(
        "tokens:    {} -> {}  ({:.0}% fewer)",
        trad.tokens,
        cm.tokens,
        pct(trad.tokens as f64, cm.tokens as f64),
    );
    println!(
        "LLM turns: {} -> {}  (each turn is a network round-trip)",
        trad.turns, cm.turns
    );
    println!(
        "latency:   {:.1}s -> {:.1}s  ({:.0}% faster)",
        trad_time.as_secs_f64(),
        cm_time.as_secs_f64(),
        pct(trad_time.as_secs_f64(), cm_time.as_secs_f64()),
    );
}
