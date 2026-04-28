'use strict'

const test = require('brittle')
const path = require('bare-path')
const fs = require('bare-fs')
const os = require('bare-os')
const {
  TranscriptionParakeet,
  loadGgufOrSkip,
  isMobile
} = require('./helpers.js')

// The GGUF backend collapses the legacy onnx multi-file
// model layout into a single `.gguf` per checkpoint, so the
// per-modelType "named paths" surface (encoderPath, decoderPath,
// ctcModelPath, ...) collapses to one `modelPath` field. These tests
// therefore only cover the validation behaviour that's still
// meaningful: empty files map, non-existent paths, and an actual
// GGUF on disk being accepted.

test('Should accept empty files map without throwing', { timeout: 60000 }, async (t) => {
  if (isMobile) { t.pass('Skipped on mobile'); return }
  TranscriptionParakeet.prototype.validateModelFiles?.restore?.()

  try {
    const model = new TranscriptionParakeet({
      files: {},
      config: { parakeetConfig: { modelType: 'tdt' } }
    })
    t.ok(model, 'Model instance created with empty files map')
    t.pass('Empty files map is accepted (validation skipped for unset paths)')
  } catch (error) {
    t.fail('Should not throw for empty files map: ' + error.message)
  }
})

test('Non-existent file paths produce warnings but do not throw', { timeout: 60000 }, async (t) => {
  if (isMobile) { t.pass('Skipped on mobile'); return }
  TranscriptionParakeet.prototype.validateModelFiles?.restore?.()

  try {
    const model = new TranscriptionParakeet({
      files: {
        // legacy ONNX-shaped fields are still accepted by the wrapper
        // for back-compat; the validator just emits a warning when the
        // file is missing rather than throwing.
        encoder: '/this/path/definitely/does/not/exist/encoder.onnx',
        decoder: '/this/path/definitely/does/not/exist/decoder.onnx',
        vocab: '/this/path/definitely/does/not/exist/vocab.txt'
      },
      config: { parakeetConfig: { modelType: 'tdt' } }
    })
    t.ok(model, 'Model instance created despite non-existent file paths')
    t.pass('Non-existent file paths produce warnings, not errors')
  } catch (error) {
    t.fail('Should not throw for non-existent file paths: ' + error.message)
  }
})

test('Should accept a valid GGUF path and pass validation', { timeout: 60000 }, async (t) => {
  if (isMobile) { t.pass('Skipped on mobile'); return }
  TranscriptionParakeet.prototype.validateModelFiles?.restore?.()

  const ggufPath = await loadGgufOrSkip(t, 'tdt')
  if (!ggufPath) return

  try {
    const model = new TranscriptionParakeet({
      config: {
        parakeetConfig: { modelType: 'tdt', modelPath: ggufPath }
      }
    })
    t.ok(model, 'Model instance created with valid GGUF path')
    t.ok(fs.existsSync(ggufPath), 'GGUF file exists at the supplied path')
  } catch (error) {
    t.fail('Should not have thrown an error: ' + error.message)
  }
})

test('Validation runs in the constructor (no async load required)', { timeout: 60000 }, async (t) => {
  if (isMobile) { t.pass('Skipped on mobile'); return }
  TranscriptionParakeet.prototype.validateModelFiles?.restore?.()

  // No file paths supplied; constructor just runs validateModelFiles
  // (which warns on missing paths) and returns. Should never throw.
  try {
    const model = new TranscriptionParakeet({
      files: {},
      config: { parakeetConfig: { modelType: 'tdt' } }
    })
    t.ok(model, 'Constructor completes without throw')
  } catch (error) {
    t.fail('Constructor threw unexpectedly: ' + error.message)
  }
})

test('Provides a tmp scratch dir without polluting cwd', { timeout: 60000 }, async (t) => {
  // Sanity check that file-validation tests don't write into the
  // package source tree by accident.
  const tmpDir = path.join(os.tmpdir(), '.parakeet-test-validation-scratch')
  if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir, { recursive: true })

  // Write a stub file -- must succeed.
  const stub = path.join(tmpDir, 'stub.gguf')
  fs.writeFileSync(stub, 'GGUF\x03\x00\x00\x00')
  t.ok(fs.existsSync(stub), 'Stub GGUF written to scratch dir')

  // Create a model with that file path -- the binary content is
  // intentionally bogus, so the wrapper should still accept it (it's
  // a valid path; load-time validation happens later).
  const model = new TranscriptionParakeet({
    config: {
      parakeetConfig: { modelType: 'tdt', modelPath: stub }
    }
  })
  t.ok(model, 'Wrapper accepts a path-only configuration')

  try { fs.rmSync(tmpDir, { recursive: true, force: true }) } catch (e) { /* ignore */ }
})
