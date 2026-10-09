//! The shared plugin tests (`dfn3-plugin/tests/conformance`) against DeepFilterNet3-LL.
use dfn3ll_ladspa::Dfn3LlNetwork as Net;
use dfn3ll_ladspa::DESCRIPTOR;

/// Measured input-to-output delay without the voice gate.
const ENGINE_DELAY: usize = 959;

#[path = "../../dfn3-plugin/tests/conformance/mod.rs"]
mod conformance;
