'use strict'

const test = require('brittle')
const path = require('bare-path')
const fs = require('bare-fs')
const os = require('bare-os')
const {
  binding,
  ParakeetInterface,
  setupJsLogger,
  getNamedPathsConfig,
  isMobile
} = require('./helpers.js')

// The GGUF backend ships a single self-contained `.gguf`
// per model. The legacy onnx version of this test created multiple
// corrupted .onnx files; we now create a single corrupted .gguf and
// expect the binding to surface a load-time error to JS via the same
// Error-event channel.

function makeTempDir (label) {
  const root = isMobile
    ? path.join(global.testDir || os.tmpdir(), '.parakeet-test-' + label)
    : path.join(os.tmpdir(), '.parakeet-test-' + label)
  if (!fs.existsSync(root)) fs.mkdirSync(root, { recursive: true })
  return root
}

function cleanupDir (dirPath) {
  if (!fs.existsSync(dirPath)) return
  try {
    fs.rmSync(dirPath, { recursive: true, force: true })
  } catch (e) {
    /* ignore */
  }
}

function writeBadGguf (dir, contents) {
  const p = path.join(dir, 'corrupted.gguf')
  fs.writeFileSync(p, contents)
  return p
}

async function expectLoadError (t, ggufPath) {
  const loggerBinding = setupJsLogger(binding)
  let errorReceived = false
  let errorMessage = ''
  let resolvePromise = null
  const waitForError = new Promise(resolve => { resolvePromise = resolve })

  function outputCallback (handle, event, id, output, error) {
    if (event === 'Error') {
      errorReceived = true
      errorMessage = error || ''
      if (resolvePromise) {
        resolvePromise()
        resolvePromise = null
      }
    }
  }

  const config = {
    modelPath: ggufPath,
    modelType: 'tdt',
    maxThreads: 4,
    useGPU: false,
    sampleRate: 16000,
    channels: 1,
    ...getNamedPathsConfig('tdt', ggufPath)
  }

  let parakeet = null
  try {
    parakeet = new ParakeetInterface(binding, config, outputCallback)
    try {
      await parakeet.activate()
    } catch (error) {
      errorReceived = true
      errorMessage = error.message
      if (resolvePromise) {
        resolvePromise()
        resolvePromise = null
      }
    }
    const timeout = setTimeout(() => {
      if (resolvePromise) {
        resolvePromise()
        resolvePromise = null
      }
    }, 5000)
    await waitForError
    clearTimeout(timeout)
  } finally {
    if (parakeet) {
      try { await parakeet.destroyInstance() } catch (e) { /* ignore */ }
    }
    try { loggerBinding.releaseLogger() } catch (e) { /* ignore */ }
  }

  t.ok(errorReceived, `Should receive Error event or exception (got "${errorMessage}")`)
}

test('Corrupted GGUF (junk bytes) should emit Error event to JavaScript', { timeout: 60000 }, async (t) => {
  const dir = makeTempDir('corrupted-models')
  try {
    const gguf = writeBadGguf(dir,
      'This is not a valid GGUF file -- the magic number GGUF should be at offset 0')
    await expectLoadError(t, gguf)
  } finally {
    cleanupDir(dir)
  }
})

test('Empty GGUF should emit Error event to JavaScript', { timeout: 60000 }, async (t) => {
  const dir = makeTempDir('empty-models')
  try {
    const gguf = writeBadGguf(dir, '')
    await expectLoadError(t, gguf)
  } finally {
    cleanupDir(dir)
  }
})

test('Truncated GGUF (correct magic, no data) should emit Error event to JavaScript', { timeout: 60000 }, async (t) => {
  const dir = makeTempDir('truncated-models')
  try {
    // 4-byte GGUF magic followed by truncated metadata. Just enough to
    // get past the initial file-existence check but fail at parse.
    const truncated = Buffer.from([
      0x47, 0x47, 0x55, 0x46,           // "GGUF" magic
      0x03, 0x00, 0x00, 0x00,           // version=3 (little-endian uint32)
      0xFF, 0xFF, 0xFF, 0xFF,           // garbage tensor count
      0xFF, 0xFF, 0xFF, 0xFF
    ])
    const gguf = writeBadGguf(dir, truncated)
    await expectLoadError(t, gguf)
  } finally {
    cleanupDir(dir)
  }
})
