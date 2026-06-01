//! Local tools written in Rust, exposed to the model the same way MCP servers
//! are. A `LocalTools` set becomes a virtual server; the model imports
//! `./servers/<name>` and calls its functions, which run your Rust closures on
//! the engine island. This lets a Rig app mix its own in-process tools with
//! remote/CLI MCP servers under one code-execution surface.

use std::collections::HashMap;
use std::future::Future;
use std::rc::Rc;

use serde_json::Value;

use crate::error::{Error, Result};
use crate::source::{LocalFuture, ToolSource};
use crate::types::{ServerInfo, ToolInfo};

/// An async handler for a local tool: JSON in, JSON out.
type Handler = Box<dyn Fn(Value) -> LocalFuture<std::result::Result<Value, String>> + Send>;

struct LocalTool {
    info: ToolInfo,
    handler: Handler,
}

/// A named group of in-Rust tools, registered on the builder via
/// [`crate::CodeModeBuilder::local_tools`].
///
/// ```
/// use codemode::LocalTools;
/// use serde_json::json;
///
/// let tools = LocalTools::new("math")
///     .tool("add", "Add two numbers", json!({
///         "type": "object",
///         "properties": { "a": { "type": "number" }, "b": { "type": "number" } },
///         "required": ["a", "b"],
///     }), |args| async move {
///         let a = args["a"].as_f64().unwrap_or(0.0);
///         let b = args["b"].as_f64().unwrap_or(0.0);
///         Ok(json!(a + b))
///     });
/// assert_eq!(tools.server(), "math");
/// ```
pub struct LocalTools {
    server: String,
    tools: Vec<LocalTool>,
}

impl std::fmt::Debug for LocalTools {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalTools")
            .field("server", &self.server)
            .field("tools", &self.tool_names())
            .finish()
    }
}

impl LocalTools {
    pub fn new(server: impl Into<String>) -> Self {
        Self {
            server: server.into(),
            tools: Vec::new(),
        }
    }

    /// The virtual server name these tools are exposed under.
    pub fn server(&self) -> &str {
        &self.server
    }

    /// Register a tool: a name, a description, a JSON Schema for the arguments,
    /// and an async handler.
    pub fn tool<F, Fut>(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
        handler: F,
    ) -> Self
    where
        F: Fn(Value) -> Fut + Send + 'static,
        Fut: Future<Output = std::result::Result<Value, String>> + 'static,
    {
        let info = ToolInfo {
            name: name.into(),
            description: Some(description.into()),
            input_schema,
        };
        let handler: Handler = Box::new(move |args| Box::pin(handler(args)));
        self.tools.push(LocalTool { info, handler });
        self
    }

    pub(crate) fn tool_names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.info.name.clone()).collect()
    }

    pub(crate) fn into_source(self) -> Rc<dyn ToolSource> {
        let tools = self
            .tools
            .into_iter()
            .map(|t| (t.info.name.clone(), t))
            .collect();
        Rc::new(LocalToolSet {
            server: self.server,
            tools,
        })
    }
}

struct LocalToolSet {
    server: String,
    tools: HashMap<String, LocalTool>,
}

impl ToolSource for LocalToolSet {
    fn server_names(&self) -> Vec<String> {
        vec![self.server.clone()]
    }

    fn servers(self: Rc<Self>) -> LocalFuture<Result<Vec<ServerInfo>>> {
        Box::pin(async move {
            Ok(vec![ServerInfo {
                name: self.server.clone(),
                description: None,
            }])
        })
    }

    fn tools(self: Rc<Self>, server: String) -> LocalFuture<Result<Vec<ToolInfo>>> {
        Box::pin(async move {
            if server != self.server {
                return Err(Error::UnknownServer(server));
            }
            Ok(self.tools.values().map(|t| t.info.clone()).collect())
        })
    }

    fn call(
        self: Rc<Self>,
        server: String,
        tool: String,
        args: Value,
    ) -> LocalFuture<Result<Value>> {
        Box::pin(async move {
            if server != self.server {
                return Err(Error::UnknownServer(server));
            }
            let fut = {
                let entry = self
                    .tools
                    .get(&tool)
                    .ok_or_else(|| Error::Tool(format!("unknown tool '{tool}'")))?;
                (entry.handler)(args)
            };
            fut.await.map_err(Error::Tool)
        })
    }
}

/// Routes calls across several sources (MCP servers + local tool sets) by
/// server name.
pub(crate) struct CompositeSource {
    sources: Vec<Rc<dyn ToolSource>>,
    /// server name -> the source that owns it, built once at construction.
    /// Server names are unique (the builder rejects duplicates), so a flat map
    /// is enough and avoids rebuilding a name list on every tool call.
    routes: HashMap<String, Rc<dyn ToolSource>>,
}

impl CompositeSource {
    pub(crate) fn new(sources: Vec<Rc<dyn ToolSource>>) -> Self {
        let mut routes = HashMap::new();
        for source in &sources {
            for name in source.server_names() {
                routes.insert(name, source.clone());
            }
        }
        Self { sources, routes }
    }

    fn route(&self, server: &str) -> Option<Rc<dyn ToolSource>> {
        self.routes.get(server).cloned()
    }
}

impl ToolSource for CompositeSource {
    fn server_names(&self) -> Vec<String> {
        self.sources.iter().flat_map(|s| s.server_names()).collect()
    }

    fn servers(self: Rc<Self>) -> LocalFuture<Result<Vec<ServerInfo>>> {
        Box::pin(async move {
            let mut all = Vec::new();
            for source in &self.sources {
                all.extend(source.clone().servers().await?);
            }
            Ok(all)
        })
    }

    fn tools(self: Rc<Self>, server: String) -> LocalFuture<Result<Vec<ToolInfo>>> {
        Box::pin(async move {
            let source = self
                .route(&server)
                .ok_or_else(|| Error::UnknownServer(server.clone()))?;
            source.tools(server).await
        })
    }

    fn call(
        self: Rc<Self>,
        server: String,
        tool: String,
        args: Value,
    ) -> LocalFuture<Result<Value>> {
        Box::pin(async move {
            let source = self
                .route(&server)
                .ok_or_else(|| Error::UnknownServer(server.clone()))?;
            source.call(server, tool, args).await
        })
    }

    fn shutdown(self: Rc<Self>) -> LocalFuture<()> {
        Box::pin(async move {
            for source in &self.sources {
                source.clone().shutdown().await;
            }
        })
    }
}
