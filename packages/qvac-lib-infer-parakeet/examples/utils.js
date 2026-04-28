'use strict'

// Shared helpers for the (flag-driven) examples. The ggml backend
// uses a single .gguf per checkpoint, so this file only deals with
// single-file GGUF loading + audio decode + a small streaming
// callback adapter.

const fs = require('bare-fs')
const path = require('bare-path')

// Mirror of qvac_lib_inference_addon_cpp::logger::Priority. The
// native binding queues every priority; we filter at WARNING+ here
// so kernel-JIT INFO spam from ggml's GPU backends never reaches the
// console. To see INFO/DEBUG, lower NATIVE_MIN_PRIORITY below.
const LOG_PRIORITIES = ['ERROR', 'WARNING', 'INFO', 'DEBUG']
const NATIVE_MIN_PRIORITY = 1 // WARNING

const JOB_TRACKER_GRACE_MS = 5000

/**
 * Install a JS-side sink for native log messages. Filters at
 * WARNING+ so ggml's metal/opencl/vulkan kernel-JIT INFO lines stay
 * silent. Edit NATIVE_MIN_PRIORITY at the top of this file to see
 * INFO / DEBUG.
 *
 * @param {Object} binding - the native binding exported from binding.js
 */
function setupLogger (binding) {
  if (binding.__qvacExampleLoggerSet) return
  binding.setLogger((priority, message) => {
    if (priority > NATIVE_MIN_PRIORITY) return
    const name = LOG_PRIORITIES[priority] || `UNKNOWN(${priority})`
    console.log(`[C++ ${name}] ${message}`)
  })
  binding.__qvacExampleLoggerSet = true
}

/**
 * Read a file using streams to handle large GGUFs (>2 GiB).
 * @param {string} filePath
 * @returns {Promise<Buffer>}
 */
function readFileAsStream (filePath) {
  return new Promise((resolve, reject) => {
    const chunks = []
    const stream = fs.createReadStream(filePath)
    stream.on('data', chunk => chunks.push(chunk))
    stream.on('end', () => resolve(Buffer.concat(chunks)))
    stream.on('error', reject)
  })
}

/**
 * Parse a WAV file (RIFF/PCM int16 mono) into a Float32Array of
 * normalised samples. Skips non-`data` chunks.
 *
 * @param {string} wavPath
 * @returns {Float32Array}
 */
function parseWavFile (wavPath) {
  const buffer = fs.readFileSync(wavPath)
  if (buffer.toString('utf8', 0, 4) !== 'RIFF') throw new Error('Not a valid WAV file')
  if (buffer.toString('utf8', 8, 12) !== 'WAVE') throw new Error('Not a valid WAV file')

  let pos = 12
  while (pos < buffer.length - 8) {
    const id = buffer.toString('utf8', pos, pos + 4)
    const sz = buffer.readUInt32LE(pos + 4)
    if (id === 'data') {
      const data = buffer.slice(pos + 8, pos + 8 + sz)
      const samples = new Float32Array(data.length / 2)
      for (let i = 0; i < samples.length; i++) {
        samples[i] = data.readInt16LE(i * 2) / 32768
      }
      return samples
    }
    pos += 8 + sz + (sz % 2)
  }
  throw new Error('No data chunk found in WAV file')
}

/**
 * Convert a raw int16 little-endian PCM buffer to a normalised
 * Float32Array. Used for `.raw` audio fixtures.
 *
 * @param {Buffer} rawBuffer
 * @returns {Float32Array}
 */
function convertRawToFloat32 (rawBuffer) {
  const view = new Int16Array(rawBuffer.buffer, rawBuffer.byteOffset, rawBuffer.length / 2)
  const out = new Float32Array(view.length)
  for (let i = 0; i < view.length; i++) out[i] = view[i] / 32768
  return out
}

/**
 * Stream a single GGUF file into the addon via the
 * `loadWeights({filename, chunk, completed})` API.
 *
 * @param {Object} parakeet - ParakeetInterface instance
 * @param {string} ggufPath - absolute path to the .gguf file
 */
async function loadModelWeights (parakeet, ggufPath) {
  if (!ggufPath.toLowerCase().endsWith('.gguf')) {
    throw new Error(`loadModelWeights: expected a .gguf path, got "${ggufPath}"`)
  }
  if (!fs.existsSync(ggufPath)) {
    throw new Error(`GGUF model not found: ${ggufPath}`)
  }
  const filename = path.basename(ggufPath)
  const buffer = await readFileAsStream(ggufPath)
  const chunk = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength)
  await parakeet.loadWeights({ filename, chunk, completed: true })
  console.log(`   Loaded: ${filename} (${(buffer.length / 1024 / 1024).toFixed(1)} MB)`)
}

/**
 * Validate that required paths exist on disk.
 * @param {Object} paths - { model: string, audio?: string }
 * @returns {boolean}
 */
function validatePaths (paths) {
  if (!fs.existsSync(paths.model)) {
    console.error(`Model not found: ${paths.model}`)
    console.error("Run 'npm run download-models' or pass --model </path/to/model.gguf>.")
    return false
  }
  if (paths.audio && !fs.existsSync(paths.audio)) {
    console.error(`Audio not found: ${paths.audio}`)
    return false
  }
  return true
}

/**
 * Create a promise that resolves when the addon emits both an Output
 * event and a JobEnded event (whichever order). Solves the race where
 * fast models can fire JobEnded before the Output payload lands.
 *
 * @returns {{ promise, resolve, transcriptions, markOutput, markJobEnded }}
 */
function createJobTracker () {
  const transcriptions = []
  let resolveJob = null
  let hasOutput = false
  let jobEnded = false
  let graceTimeout = null
  const promise = new Promise(resolve => { resolveJob = resolve })

  const tryResolve = () => {
    if (hasOutput && jobEnded) {
      if (graceTimeout) clearTimeout(graceTimeout)
      resolveJob()
    }
  }

  return {
    promise,
    resolve: () => resolveJob(),
    transcriptions,
    markOutput () { hasOutput = true; tryResolve() },
    markJobEnded () {
      jobEnded = true
      tryResolve()
      if (!hasOutput) {
        graceTimeout = setTimeout(() => resolveJob(), JOB_TRACKER_GRACE_MS)
      }
    }
  }
}

/**
 * Create a standard output callback that pushes per-segment transcripts
 * into a tracker, optionally streaming them to stdout.
 *
 * @param {Object} tracker - from createJobTracker()
 * @param {Object} [options] - { verbose: boolean }
 */
function createOutputCallback (tracker, { verbose = false } = {}) {
  return (handle, event, id, output, error) => {
    if (error) {
      console.error('Error:', error)
      return
    }
    if (event === 'Output' && output) {
      const segments = Array.isArray(output) ? output : [output]
      for (const seg of segments) {
        if (!seg || !seg.text || !seg.toAppend) continue
        tracker.transcriptions.push(seg)
        if (verbose) {
          const a = seg.start?.toFixed(2) ?? '?'
          const b = seg.end?.toFixed(2) ?? '?'
          console.log(`   [${a}s - ${b}s] ${seg.text}`)
        }
      }
      tracker.markOutput()
    }
    if (event === 'JobEnded') tracker.markJobEnded()
  }
}

/**
 * Print transcription results to stdout in a uniform "=== RESULT ==="
 * banner block.
 */
function printResults (transcriptions) {
  console.log('\n=== RESULT ===')
  console.log('='.repeat(50))
  if (transcriptions.length > 0) {
    const text = transcriptions.map(s => s.text).join(' ').trim().replace(/\s+/g, ' ')
    console.log(text)
  } else {
    console.log('[No speech detected]')
  }
  console.log('='.repeat(50))
}

module.exports = {
  setupLogger,
  readFileAsStream,
  parseWavFile,
  convertRawToFloat32,
  loadModelWeights,
  validatePaths,
  createJobTracker,
  createOutputCallback,
  printResults
}
