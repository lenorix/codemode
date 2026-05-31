//! The internal seam to "where tools come from". The real implementation is the
//! MCP client (`mcp`); tests use an in-memory fake. Methods take `Rc<Self>` and
//! return `'static`, non-`Send` futures so they compose with the Boa bridge,
//! which lives on the single-threaded engine island.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

use serde_json::Value;

use crate::error::Result;
use crate::types::{ServerInfo, ToolInfo};

pub type LocalFuture<T> = Pin<Box<dyn Future<Output = T>>>;

pub trait ToolSource {
    /// The server names this source provides, for routing in a composite.
    fn server_names(&self) -> Vec<String>;

    fn servers(self: Rc<Self>) -> LocalFuture<Result<Vec<ServerInfo>>>;
    fn tools(self: Rc<Self>, server: String) -> LocalFuture<Result<Vec<ToolInfo>>>;
    fn call(
        self: Rc<Self>,
        server: String,
        tool: String,
        args: Value,
    ) -> LocalFuture<Result<Value>>;

    /// Release resources (e.g. cancel MCP connections) while the tokio reactor
    /// is still alive. Called on the island before it winds down.
    fn shutdown(self: Rc<Self>) -> LocalFuture<()> {
        Box::pin(async {})
    }
}

/// Deny-by-default set of `{server, tool}` pairs reachable in a run.
#[derive(Debug, Default, Clone)]
pub struct AllowList {
    servers: HashMap<String, Allowed>,
}

#[derive(Debug, Clone)]
enum Allowed {
    All,
    Only(HashSet<String>),
}

impl AllowList {
    pub fn allow(
        &mut self,
        server: impl Into<String>,
        tools: impl IntoIterator<Item = impl Into<String>>,
    ) {
        let set = tools.into_iter().map(Into::into).collect();
        self.servers.insert(server.into(), Allowed::Only(set));
    }

    pub fn allow_all(&mut self, server: impl Into<String>) {
        self.servers.insert(server.into(), Allowed::All);
    }

    pub fn allows(&self, server: &str, tool: &str) -> bool {
        match self.servers.get(server) {
            Some(Allowed::All) => true,
            Some(Allowed::Only(set)) => set.contains(tool),
            None => false,
        }
    }

    pub fn server_allowed(&self, server: &str) -> bool {
        self.servers.contains_key(server)
    }
}

#[cfg(test)]
pub(crate) mod fake {
    //! A canned, in-memory `ToolSource` for deterministic tests.

    use super::*;

    /// (server, tool, args) -> a canned result.
    pub(crate) type Responder = Box<dyn Fn(&str, &str, &Value) -> Value>;

    pub(crate) struct FakeSource {
        pub servers: Vec<ServerInfo>,
        pub tools: HashMap<String, Vec<ToolInfo>>,
        pub responder: Responder,
        /// Counts `tools()` calls, so tests can assert schema caching.
        pub tools_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl ToolSource for FakeSource {
        fn server_names(&self) -> Vec<String> {
            self.servers.iter().map(|s| s.name.clone()).collect()
        }

        fn servers(self: Rc<Self>) -> LocalFuture<Result<Vec<ServerInfo>>> {
            Box::pin(async move { Ok(self.servers.clone()) })
        }

        fn tools(self: Rc<Self>, server: String) -> LocalFuture<Result<Vec<ToolInfo>>> {
            self.tools_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Box::pin(async move {
                match self.tools.get(&server).cloned() {
                    Some(tools) => Ok(tools),
                    None => Err(crate::error::Error::UnknownServer(server)),
                }
            })
        }

        fn call(
            self: Rc<Self>,
            server: String,
            tool: String,
            args: Value,
        ) -> LocalFuture<Result<Value>> {
            Box::pin(async move { Ok((self.responder)(&server, &tool, &args)) })
        }
    }
}
