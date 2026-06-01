//! Surface B in a few lines: drop codemode into a Rig agent.
//!
//! This is the ergonomics demo, the integration the README promises. We build a
//! `CodeMode` over a small in-Rust "store" server, then call `.code_mode(&cm)` on
//! a Rig `AgentBuilder`. That single call registers the discover / find / execute
//! tools and appends the usage guidance, so the model answers by writing one JS
//! program instead of chaining tool calls. There is no measurement here on
//! purpose; if you want the token and latency numbers behind code mode, that's
//! `token_savings.rs`, which hand-rolls its own loop so it can count them.
//!
//! Runs against a local OpenAI-compatible server (e.g. LM Studio) at
//! 127.0.0.1:1234. The `rig-example` feature adds rig-core's openai provider so
//! this example can reach the server (the library `rig` feature is just the
//! adapter). Start the server, then:
//!
//!   cargo run --example rig_agent --features rig-example

use std::sync::{Arc, LazyLock};

use codemode::rig::CodeModeExt;
use codemode::{CodeMode, LocalTools};
use rig_core::client::completion::CompletionClient;
use rig_core::completion::Prompt;
use rig_core::providers::openai;
use serde_json::{Value, json};

// LM Studio ignores the key but the client requires one.
const BASE_URL: &str = "http://127.0.0.1:1234/v1";
const API_KEY: &str = "lm-studio";
const MODEL: &str = "qwen/qwen3.5-9b";

const TASK: &str = "Produce a top-sellers report. Among products whose average review rating is at least \
    4.5 and that are currently in stock, list the top three by 30-day units sold, best first, each \
    formatted as \"<name> - <units> units at $<price>\" (price in whole dollars).";

// --- The "store" dataset and its tool handlers (see examples/catalog.json) ---

static CATALOG: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("catalog.json")).expect("catalog.json is valid JSON")
});

fn products() -> &'static [Value] {
    CATALOG["products"].as_array().expect("products array")
}

fn product(id: &str) -> Option<&'static Value> {
    products().iter().find(|p| p["id"] == id)
}

fn list_products() -> Value {
    Value::Array(
        products()
            .iter()
            .map(|p| json!({ "id": p["id"], "name": p["name"], "category": p["category"] }))
            .collect(),
    )
}

fn get_reviews(id: &str) -> Value {
    product(id)
        .map(|p| p["reviews"].clone())
        .unwrap_or_else(|| json!([]))
}

fn get_inventory(id: &str) -> Value {
    match product(id) {
        Some(p) => json!({ "in_stock": p["in_stock"], "restock_days": p["restock_days"] }),
        None => json!({ "error": "unknown product" }),
    }
}

fn get_pricing(id: &str) -> Value {
    match product(id) {
        Some(p) => json!({ "price_cents": p["price_cents"], "currency": "USD" }),
        None => json!({ "error": "unknown product" }),
    }
}

fn get_sales(id: &str) -> Value {
    match product(id) {
        Some(p) => json!({ "units_sold_30d": p["units_sold_30d"] }),
        None => json!({ "error": "unknown product" }),
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let by_product =
        json!({ "type": "object", "properties": { "product_id": { "type": "string" } } });

    let cm =
        Arc::new(
            CodeMode::builder()
                .local_tools(
                    LocalTools::new("store")
                        .tool(
                            "listProducts",
                            "List all products.",
                            json!({ "type": "object" }),
                            |_args| async move { Ok(list_products()) },
                        )
                        .tool(
                            "getReviews",
                            "Get reviews for a product.",
                            by_product.clone(),
                            |args| async move {
                                Ok(get_reviews(args["product_id"].as_str().unwrap_or("")))
                            },
                        )
                        .tool(
                            "getInventory",
                            "Get stock status for a product.",
                            by_product.clone(),
                            |args| async move {
                                Ok(get_inventory(args["product_id"].as_str().unwrap_or("")))
                            },
                        )
                        .tool(
                            "getPricing",
                            "Get pricing for a product.",
                            by_product.clone(),
                            |args| async move {
                                Ok(get_pricing(args["product_id"].as_str().unwrap_or("")))
                            },
                        )
                        .tool(
                            "getSales",
                            "Get recent sales for a product.",
                            by_product,
                            |args| async move {
                                Ok(get_sales(args["product_id"].as_str().unwrap_or("")))
                            },
                        ),
                )
                .build()
                .await
                .expect("build code mode"),
        );

    // LM Studio speaks the Chat Completions API; the default client targets the
    // newer Responses API, so switch it over with `.completions_api()`.
    let client = openai::Client::builder()
        .api_key(API_KEY)
        .base_url(BASE_URL)
        .build()
        .expect("build openai client")
        .completions_api();

    let agent = client
        .agent(MODEL)
        .preamble("You are a data assistant for an online store.")
        .code_mode(&cm)
        .build();

    println!("Task: {TASK}\n");

    // multi-turn (via max_turns) lets the model call `execute` and then answer
    // in a follow-up turn; a single turn would stop after the tool call.
    let answer = agent
        .prompt(TASK)
        .max_turns(10)
        .await
        .expect("agent prompt failed");

    println!("Answer:\n{answer}");
}
