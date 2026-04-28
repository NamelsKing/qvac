// Pure-ggml backend (sourced from qvac-parakeet.cpp). The previous binding required
// `@qvac/onnx` to surface the bundled onnxruntime.bare module before the
// addon binary loaded; the new backend ships its own ggml runtime
// statically linked into the addon, so no peer module is needed.

module.exports = require.addon()
