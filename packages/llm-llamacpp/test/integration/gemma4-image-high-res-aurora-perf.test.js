'use strict'
// QVAC-18298: Gemma4-VL high-res-aurora-image perf row. One image per file
// (like the SmolVLM2 image-*.test.js) so each Device Farm test stays under
// the 30-minute mobile cap. Asserts an aurora keyword + records perf.

const test = require('brittle')
const { GEMMA4_MODEL, IMAGE_CASES, isBenchmarkRun, runVlmImagePerf } = require('./_vlm-image-perf.js')

// QVAC-19368: aurora is the heaviest image; skip it on normal on-PR runs
// (elephant + fruit-plate still run) so the Android leg stays under ~1h.
// The benchmark (QVAC_PERF_RUNS>1) runs all 3 images.
test('Gemma4-VL image perf [high-res aurora]', { timeout: 1_800_000, skip: !isBenchmarkRun }, async t => {
  await runVlmImagePerf(t, GEMMA4_MODEL, IMAGE_CASES['high-res-aurora'])
})

setImmediate(() => {
  setTimeout(() => {}, 500)
})
