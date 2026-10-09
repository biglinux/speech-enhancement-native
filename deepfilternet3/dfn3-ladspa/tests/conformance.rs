//! The shared plugin tests (`dfn3-plugin/tests/conformance`) against DeepFilterNet3.
use dfn3_ladspa::Dfn3Network as Net;
use dfn3_ladspa::DESCRIPTOR;

/// Measured input-to-output delay without the voice gate.
const ENGINE_DELAY: usize = 1919;

#[path = "../../dfn3-plugin/tests/conformance/mod.rs"]
mod conformance;
