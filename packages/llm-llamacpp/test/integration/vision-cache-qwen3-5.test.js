'use strict'
// QVAC-19118 (A2): vision prefix cache integration tests for Qwen3.5-0.8B.
// Qwen's VLM uses M-RoPE, so the cached embedding's nPos differs from nTokens —
// this exercises the VisionCacheEntry nPos path that Gemma's standard RoPE does
// not. Split into its own file so each model loads in an isolated bare process —
// see _vision-cache-common.js.

const { runVisionCacheTests } = require('./_vision-cache-common.js')

const QWEN3_5 = {
  label: 'Qwen3.5-0.8B',
  llmModel: {
    modelName: 'Qwen3.5-0.8B-Q8_0.gguf',
    downloadUrl: 'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf'
  },
  projModel: {
    modelName: 'mmproj-Qwen3.5-0.8B-F16.gguf',
    downloadUrl: 'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/mmproj-F16.gguf'
  },
  // Vision config from qwen3-5.test.js's image test.
  visionConfig: {
    gpu_layers: '98',
    ctx_size: '4096',
    temp: '0',
    seed: '42',
    verbosity: '2'
  },
  // 0.8B embeddings are smaller (~1 MB/entry), so a 2 MB budget holds one image
  // but not two → the second distinct image forces an LRU eviction.
  evictBudgetMb: '2'
}

runVisionCacheTests(QWEN3_5)

setImmediate(() => {
  setTimeout(() => {}, 500)
})
