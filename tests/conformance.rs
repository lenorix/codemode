//! The reference Boa backend passes the conformance suite (the cross-backend
//! contract, defined in `tests/common/`). It's deterministic, no LLM, so it
//! runs by default with `cargo test`.
//!
//! A new backend proves itself the same way: implement `CodeRuntime`, supply the
//! source for each `ConformanceProgram` (JS backends reuse `javascript_programs()`),
//! and run it through `common::assert_js_conformant` (or `common::check`).

mod common;

#[test]
fn boa_passes_the_conformance_suite() {
    common::assert_js_conformant(codemode::Boa::new());
}
