
## References

Based on:
- https://blog.cloudflare.com/code-mode/
- https://www.anthropic.com/engineering/code-execution-with-mcp

## Summary

Instead to let the harness to expose the MCP client tools directly, expose 3 tools to discover MCP servers,
 find tools and execute TypeScript code to use the tools. Then, it's run in a restricted environment, and call
 to tools bind to code that handles the MCP tool execution.

## This Implementation

The idea is to use deno/bun embeddings to run expose functions that will handle calls to the MCP tools, restricting the rest.

In that way, deno/bun run TypeScript, but it's limited to only the exposed functions. Without networking, filesystem, etc.

```text
TypeScript
 ↓
Rust
 ↓
MCP
```

```typescript
import * as gdrive from './servers/google-drive';
import * as fs from './servers/filesystem';
export async function saveSheetAsCsv(sheetId: string) {
  const data = await gdrive.getSheet({ sheetId });
  const csv = data.map(row => row.join(',')).join('\n');
  await fs.writeFile(`./workspace/sheet-${sheetId}.csv`, csv);
  return `./workspace/sheet-${sheetId}.csv`;
}
```

## Other Resources

- https://deno.com/blog/roll-your-own-javascript-runtime
- https://gist.github.com/andelf/61574d03353998a7b16a358a6fd5a097
