# codemode-mcp

Run MCP tools by writing code instead of calling them one by one.

The normal way to use MCP is to hand the model every tool a server exposes. It works, but it gets
expensive fast. All those tool definitions sit in the context, and every intermediate result has to
travel back through the model before you can do anything with it.

The idea here, borrowed from Cloudflare's "code mode" and Anthropic's writeup, is to flip that
around. Give the model just three tools (discover servers, find a server's tools, and execute code)
and let it write a program that does the orchestration. Loops, filtering, chaining calls together,
all of it happens in the code instead of the chat, so intermediate results never re-enter the
context.

The code runs in an embedded [Boa](https://github.com/boa-dev/boa) engine (a pure-Rust JavaScript
interpreter), locked down so it can't touch the network, the filesystem, or anything else on the
host. The only way out is the per-server modules we generate from the tools you expose, and the
allowlist behind them is enforced in Rust. The engine sits behind a `CodeRuntime` trait, so the
community can add other engines and languages later.

## Two ways to use it

As a **library / Rig dependency** (the `codemode()`-for-Rust shape): wrap your MCP servers (and/or
in-Rust tools) and give a Rig agent the three-tool surface instead of every tool.

```rust
use std::sync::Arc;
use codemode::{CodeMode, ServerConfig};
use codemode::rig::CodeModeExt;

let cm = Arc::new(
    CodeMode::builder()
        .server(ServerConfig::stdio("filesystem", "npx",
                ["-y", "@modelcontextprotocol/server-filesystem", "."]))
        .build()
        .await?,
);

let agent = client.agent(model).preamble("…").code_mode(&cm).build();
```

> Expose only the tools you trust the model to compose: isolating compute is not the boundary, the
> exposed tools are, so the model's code can chain any tools you allow (a read tool plus a write/send
> tool can exfiltrate). Keep the allowlist least-privilege. codemode runs the model's code in-process
> and is meant for trusted or steered input; if a host may feed untrusted or prompt-injected input,
> isolate the whole process at the OS level yourself (a container with cgroup CPU and memory limits).
> See [security](docs/security.md).

As a **standalone MCP server** any host can configure:

```jsonc
{ "mcpServers": { "codemode": {
    "command": "codemode-mcp",
    "args": ["serve", "--config", "servers.toml"] } } }
```

`--config` takes our `servers.toml` (servers + allowlist + limits) or a standard `.mcp.json`.

## Try it

```sh
cargo test # the full suite
cargo run --bin codemode-mcp -- tools --config servers.toml   # list the exposed servers and tools
cargo run --example token_savings # token usage: traditional vs. code mode (needs a local LLM)
cargo run --example rig_agent --features rig-example # the same task through a Rig agent (needs a local LLM)
```

The `rig_agent` example is the Surface B story end to end: a few lines wire `CodeMode` into a Rig
agent with `.code_mode(&cm)`. The `rig` feature is just the adapter (you bring your own provider);
`rig-example` additionally pulls rig-core's openai provider so the example can reach a local server.

The `token_savings` example runs the same task both ways against a local OpenAI-compatible model and
reports, for each, the tokens, the number of LLM turns, and the wall-clock time. The task is a
production-shaped report (a "top sellers among well-reviewed, in-stock products" query that chains
five tools per product). A sample run reached the same answer both ways:

```
== Result (traditional -> code mode) ==
tokens:    11950 -> 2649  (78% fewer)
LLM turns: 6 -> 2  (each turn is a network round-trip)
latency:   205.6s -> 87.9s  (57% faster)
```

The win grows with the data: in traditional tool-calling every intermediate result re-enters the
model's context and is re-tokenized each turn, while in code mode it stays in the sandbox and only
the final answer returns.

## Docs

Design and contracts live in [docs/](docs/README.md): the [Boa engine](docs/boa-integration.md),
the [engine-agnostic seam](docs/engine-agnostic.md), the [MCP client](docs/mcp-client.md), the
[two surfaces](docs/surfaces.md), the [security model](docs/security.md), and the API/CLI specs in
[docs/spec/](docs/spec/).
