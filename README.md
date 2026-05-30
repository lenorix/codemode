# code-execution-with-mcp

An experiment in running MCP tools by writing code instead of calling them one by one.

The normal way to use MCP is to hand the model every tool a server exposes. It works, but it
gets expensive fast. All those tool definitions sit in the context, and every intermediate
result has to travel back through the model before you can do anything with it.

The idea here, borrowed from Cloudflare's "code mode" and Anthropic's writeup, is to flip that
around. Give the model just three tools (one to discover servers, one to look up the tools on
them, and one to run TypeScript) and let it write code that does the orchestration. Loops,
filtering, chaining calls together, all of it happens in the code instead of the chat.

The TypeScript runs in an embedded Deno/Bun runtime, locked down so it can't touch the network,
the filesystem, or anything else on the host. The only things the code can reach are the
functions that actually call the MCP tools. Everything else is off the table.

## Status

Very early. There's basically nothing here yet beyond the scaffolding. Design notes and the
links I'm working from live in [docs/](docs/README.md).
