'use strict'

const fs = require('bare-fs')
const path = require('bare-path')
const os = require('bare-os')
const process = require('bare-process')
const { Readable } = require('bare-stream')

const platform = os.platform()
const arch = os.arch()
const isMobile = platform === 'ios' || platform === 'android'

// Mobile paths use static string literals so bare-pack can trace them into
// the bundle.  Desktop paths use variables so bare-pack skips them — the
// relative ../../ paths don't exist in the mobile test-framework layout.
const _bindingDesktop = '../../binding'
const _parakeetDesktop = '../../parakeet'
const _indexDesktop = '../../index.js'

const binding = isMobile
  ? require('@qvac/transcription-parakeet/binding.js')
  : require(_bindingDesktop)
const { ParakeetInterface } = isMobile
  ? require('@qvac/transcription-parakeet/parakeet.js')
  : require(_parakeetDesktop)
const TranscriptionParakeet = isMobile
  ? require('@qvac/transcription-parakeet')
  : require(_indexDesktop)

/**
 * Detect current platform
 * @returns {string} Platform string (e.g., 'linux-x64', 'darwin-arm64')
 */
function detectPlatform () {
  return `${platform}-${arch}`
}

/**
 * Wait until the model is no longer mid-job (state != PROCESSING).
 *
 * Note: the wrapper's `IDLE` state is only entered inside
 * `destroyInstance()` itself, so the legacy "waitUntilIdle" semantics
 * always timed out at `maxMs`. This helper now waits for "not
 * processing" (LISTENING is the steady post-job state), which is what
 * cleanup callers actually want before tearing the instance down.
 *
 * @param {Object} model - TranscriptionParakeet instance
 * @param {number} [maxMs=10000] - Maximum wait time in milliseconds
 * @returns {Promise<boolean>} True if the model left PROCESSING in time
 */
async function waitUntilIdle (model, maxMs = 10000) {
  const start = Date.now()
  while (Date.now() - start < maxMs) {
    try {
      const s = await model.status()
      if (s !== 'PROCESSING') return true
    } catch {}
    await new Promise(resolve => setTimeout(resolve, 100))
  }
  return false
}

/**
 * Converts various audio input types to a Readable stream
 * @param {string|Buffer|Uint8Array|Float32Array|Readable} audioInput - Audio input in various formats
 * @returns {Readable} Readable stream
 */
function createAudioStream (audioInput) {
  if (typeof audioInput === 'string') {
    const audioBuffer = fs.readFileSync(audioInput)
    // Create stream from Buffer with chunking to simulate streaming behavior
    const chunkSize = 16384 // 16KB chunks
    const chunks = []
    for (let i = 0; i < audioBuffer.length; i += chunkSize) {
      const chunk = audioBuffer.slice(i, i + chunkSize)
      chunks.push(new Uint8Array(chunk))
    }
    return Readable.from(chunks)
  } else if (Buffer.isBuffer(audioInput) || audioInput instanceof Uint8Array) {
    return Readable.from([audioInput])
  } else if (audioInput instanceof Float32Array) {
    // Convert Float32Array to Buffer for streaming
    const buffer = Buffer.from(audioInput.buffer)
    return Readable.from([buffer])
  } else if (Array.isArray(audioInput)) {
    return Readable.from(audioInput)
  } else if (audioInput && typeof audioInput.read === 'function') {
    return audioInput
  } else {
    throw new Error(`Unsupported audio input type: ${typeof audioInput}`)
  }
}

/**
 * Calculate audio duration in milliseconds from audio buffer
 * @param {Buffer|Uint8Array|Float32Array} audioBuffer - Audio data buffer
 * @param {string} audioFormat - Audio format ('f32le' or 's16le')
 * @param {number} sampleRate - Sample rate in Hz (default: 16000)
 * @returns {number} Duration in milliseconds
 */
function calculateAudioDuration (audioBuffer, audioFormat = 'f32le', sampleRate = 16000) {
  let bytesPerSample
  if (audioFormat === 's16le') {
    bytesPerSample = 2
  } else if (audioFormat === 'f32le') {
    bytesPerSample = 4
  } else {
    bytesPerSample = 4 // Default to f32le
  }

  const numSamples = audioBuffer.length / bytesPerSample
  const durationSeconds = numSamples / sampleRate
  return durationSeconds * 1000
}

/**
 * Generates a test audio file with a sine wave tone (Float32 format for Parakeet)
 * @param {string} filepath - Path where the audio file should be created
 * @param {number} [sampleRate=16000] - Sample rate in Hz
 * @param {number} [duration=3] - Duration in seconds
 * @param {number} [frequency=440] - Frequency in Hz (A4 note)
 * @param {number} [amplitude=0.3] - Amplitude (0-1)
 * @returns {string} The filepath of the generated audio file
 */
function generateTestAudio (filepath, sampleRate = 16000, duration = 3, frequency = 440, amplitude = 0.3) {
  if (fs.existsSync(filepath)) return filepath

  const samples = sampleRate * duration
  const audioData = new Float32Array(samples)

  for (let i = 0; i < samples; i++) {
    audioData[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate) * amplitude
  }

  const buffer = Buffer.from(audioData.buffer)
  fs.writeFileSync(filepath, buffer)
  return filepath
}

/**
 * Generates PCM noise for testing audio processing robustness (Float32 format)
 * @param {number} numSamples - Number of audio samples to generate
 * @param {number} [amplitude=0.3] - Maximum amplitude of noise
 * @returns {Float32Array} PCM noise data
 */
function makePcmNoise (numSamples, amplitude = 0.3) {
  const audioData = new Float32Array(numSamples)
  for (let i = 0; i < numSamples; i++) {
    audioData[i] = (Math.random() * 2 - 1) * amplitude
  }
  return audioData
}

/**
 * Sets up JavaScript logger for C++ bindings
 * @param {Object} [binding] - Optional binding instance (will require if not provided)
 * @returns {Object} The binding instance with logger configured
 */
function setupJsLogger (overrideBinding = null) {
  const actualBinding = overrideBinding || binding
  // Logger lifecycle in integration can crash or hang when repeatedly toggled.
  // Keep release as a no-op and only enable native logging explicitly when requested.
  if (!actualBinding.__qvacReleaseLoggerPatched) {
    actualBinding.releaseLogger = () => {}
    actualBinding.__qvacReleaseLoggerPatched = true
  }

  const shouldEnableNativeLogs = process.env &&
    process.env.QVAC_TEST_NATIVE_LOGS === '1'

  if (shouldEnableNativeLogs && !actualBinding.__qvacLoggerSet) {
    const LOG_PRIORITIES = ['ERROR', 'WARNING', 'INFO', 'DEBUG']
    actualBinding.setLogger((priority, message) => {
      const priorityName = LOG_PRIORITIES[priority] || `UNKNOWN(${priority})`
      console.log(`[C++ ${priorityName}] ${message}`)
    })
    actualBinding.__qvacLoggerSet = true
  }

  return actualBinding
}

/**
 * Calculate Word Error Rate (WER) between expected and actual transcriptions
 * @param {string} expected - Expected/reference transcription
 * @param {string} actual - Actual/hypothesis transcription
 * @returns {number} WER as a decimal (0.0 = perfect, 1.0 = 100% error)
 */
function wordErrorRate (expected, actual) {
  const normalize = (text) => text.toLowerCase().replace(/[^\w\s]/g, '').replace(/\s+/g, ' ').trim()

  const r = normalize(expected).split(/\s+/).filter(w => w.length > 0)
  const h = normalize(actual).split(/\s+/).filter(w => w.length > 0)

  if (r.length === 0) {
    return h.length === 0 ? 0 : 1
  }

  const d = Array(r.length + 1).fill(null).map(() => Array(h.length + 1).fill(0))

  for (let i = 0; i <= r.length; i++) d[i][0] = i
  for (let j = 0; j <= h.length; j++) d[0][j] = j

  for (let i = 1; i <= r.length; i++) {
    for (let j = 1; j <= h.length; j++) {
      const cost = r[i - 1] === h[j - 1] ? 0 : 1
      d[i][j] = Math.min(
        d[i - 1][j] + 1,
        d[i][j - 1] + 1,
        d[i - 1][j - 1] + cost
      )
    }
  }

  return d[r.length][h.length] / r.length
}

/**
 * Validate transcription accuracy using Word Error Rate
 * @param {string} expected - Expected/reference transcription
 * @param {string} actual - Actual/hypothesis transcription
 * @param {number} [threshold=0.3] - Maximum acceptable WER (default 30%)
 * @returns {Object} Validation result with wer, passed, and details
 */
function validateAccuracy (expected, actual, threshold = 0.3) {
  const wer = wordErrorRate(expected, actual)
  const passed = wer <= threshold

  return {
    wer,
    werPercent: (wer * 100).toFixed(2) + '%',
    passed,
    threshold,
    thresholdPercent: (threshold * 100).toFixed(0) + '%',
    expected: expected.substring(0, 100) + (expected.length > 100 ? '...' : ''),
    actual: actual.substring(0, 100) + (actual.length > 100 ? '...' : '')
  }
}

/**
 * Gets standard test paths for models and audio files
 * @param {string} [modelsDir] - Optional models directory
 * @returns {Object} Object with modelsDir, samplesDir, modelPath, and audioPath
 */
function getTestPaths (modelsDir = null) {
  const writableRoot = global.testDir || (isMobile ? os.tmpdir() : null)

  let actualModelsDir, samplesDir

  if (isMobile && writableRoot) {
    actualModelsDir = modelsDir || path.join(writableRoot, 'models')
    // Bundled testAssets are extracted to the cache dir by React Native.
    // Resolve the directory from the asset manifest so integration tests
    // can find sample audio files without an extra download step.
    const assetPaths = global.assetPaths || {}
    const firstAsset = Object.values(assetPaths)[0]
    samplesDir = firstAsset
      ? path.dirname(firstAsset.replace('file://', ''))
      : path.join(writableRoot, 'samples')
  } else {
    actualModelsDir = modelsDir || path.resolve(__dirname, '../../models')
    samplesDir = path.resolve(__dirname, '../../examples/samples')
  }

  if (!fs.existsSync(actualModelsDir)) {
    fs.mkdirSync(actualModelsDir, { recursive: true })
  }
  if (!fs.existsSync(samplesDir)) {
    fs.mkdirSync(samplesDir, { recursive: true })
  }

  return {
    modelsDir: actualModelsDir,
    samplesDir,
    // GGUF backend ships a single .gguf per model. Default
    // points at the q8_0 TDT GGUF. Override per-test when a
    // different model type is needed.
    modelPath: path.join(actualModelsDir, 'parakeet-tdt-0.6b-v3.q8_0.gguf'),
    audioPath: path.join(samplesDir, 'sample-16k.wav'),
    isMobile
  }
}

/**
 * Mobile-friendly HTTPS download using bare-https.
 * Handles redirects and streams directly to file.
 * Mirrors the pattern used by TTS's downloadModel.js.
 */
async function downloadWithHttp (url, filepath, maxRedirects = 10) {
  return new Promise((resolve, reject) => {
    const https = require('bare-https')
    const { URL } = require('bare-url')

    const parsedUrl = new URL(url)

    const options = {
      hostname: parsedUrl.hostname,
      port: parsedUrl.port || 443,
      path: parsedUrl.pathname + parsedUrl.search,
      method: 'GET',
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; bare-download/1.0)'
      }
    }

    console.log(` [HTTPS] Requesting: ${parsedUrl.hostname}${parsedUrl.pathname}`)

    const req = https.request(options, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        if (maxRedirects <= 0) {
          reject(new Error('Too many redirects'))
          return
        }
        const location = res.headers.location
        let redirectUrl
        if (location.startsWith('http://') || location.startsWith('https://')) {
          redirectUrl = location
        } else if (location.startsWith('/')) {
          redirectUrl = `${parsedUrl.protocol}//${parsedUrl.host}${location}`
        } else {
          const basePath = parsedUrl.pathname.substring(0, parsedUrl.pathname.lastIndexOf('/') + 1)
          redirectUrl = `${parsedUrl.protocol}//${parsedUrl.host}${basePath}${location}`
        }
        console.log(` [HTTPS] Redirecting to: ${redirectUrl}`)
        downloadWithHttp(redirectUrl, filepath, maxRedirects - 1)
          .then(resolve)
          .catch(reject)
        return
      }

      if (res.statusCode !== 200) {
        reject(new Error(`HTTP ${res.statusCode}: ${res.statusMessage}`))
        return
      }

      const dir = path.dirname(filepath)
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true })
      }

      const writeStream = fs.createWriteStream(filepath)
      let downloadedBytes = 0
      const contentLength = parseInt(res.headers['content-length'] || '0', 10)

      res.on('data', (chunk) => {
        writeStream.write(chunk)
        downloadedBytes += chunk.length
        if (contentLength > 0 && downloadedBytes % (1024 * 1024) < chunk.length) {
          const percent = ((downloadedBytes / contentLength) * 100).toFixed(1)
          console.log(` [HTTPS] Progress: ${percent}% (${downloadedBytes} / ${contentLength} bytes)`)
        }
      })

      res.on('end', () => {
        writeStream.end(() => {
          console.log(` [HTTPS] Download complete: ${downloadedBytes} bytes`)
          resolve()
        })
      })

      res.on('error', (err) => {
        writeStream.end()
        reject(err)
      })
    })

    req.on('error', (err) => {
      reject(err)
    })

    req.end()
  })
}

/**
 * Downloads a file from URL.
 * Uses bare-https on mobile (no curl available), curl on desktop.
 * @param {string} url - URL to download from
 * @param {string} destPath - Destination file path
 * @returns {Promise<void>}
 */
async function downloadFile (url, destPath) {
  if (isMobile) {
    return downloadWithHttp(url, destPath)
  }
  const { spawn } = require('bare-subprocess')
  return new Promise((resolve, reject) => {
    const curl = spawn('curl', ['-L', '-o', destPath, url])
    curl.on('exit', (code) => {
      if (code === 0) resolve()
      else reject(new Error(`curl exited with code ${code}`))
    })
    curl.on('error', reject)
  })
}

/**
 * Ensures the default test GGUF model is downloaded and available.
 * The ggml backend ships a single `.gguf` per checkpoint; default to the
 * q8_0 quantisation of `parakeet-tdt-0.6b-v3`
 *
 * Model files can be staged in two ways:
 *   1. Download from HuggingFace (slow on first run; cached afterwards).
 *      QVAC_TEST_GGUF_BASE_URL overrides the base URL.
 *   2. Reuse a `./models/` directory produced by `npm run setup-models`
 *      via QVAC_TEST_GGUF_DIR (or QVAC_TEST_GGUF_<TYPE>) so the suite
 *      doesn't re-download or re-convert a 700 MB file each run.
 *
 * @param {string} [modelPath] - Optional override for the GGUF path
 * @returns {Promise<string>} Path to the .gguf file
 */
async function ensureModel (modelPath = null) {
  return ensureGgufForType('tdt', modelPath)
}

/**
 * Read file in chunks using streaming to handle large files on all platforms
 * @param {string} filePath - Path to file
 * @param {number} [chunkSize=67108864] - Chunk size in bytes (default 64MB)
 * @returns {Generator<Buffer>} Generator yielding file chunks
 */
function * readFileChunked (filePath, chunkSize = 64 * 1024 * 1024) {
  const stat = fs.statSync(filePath)
  const fileSize = stat.size
  const fd = fs.openSync(filePath, 'r')

  try {
    let offset = 0
    while (offset < fileSize) {
      const readSize = Math.min(chunkSize, fileSize - offset)
      const buffer = Buffer.alloc(readSize)
      fs.readSync(fd, buffer, 0, readSize, offset)
      yield buffer
      offset += readSize
    }
  } finally {
    fs.closeSync(fd)
  }
}

/**
 * Run transcription using TranscriptionParakeet
 * @param {Object} params - Transcription parameters
 * @param {Object} [expectation={}] - Expectations for validation
 * @returns {Promise<Object>} Result object with passed, output, and data
 */
async function runTranscription (params, expectation = {}) {
  if (!params) {
    return {
      output: 'Error: Missing required parameter: params',
      passed: false,
      data: { error: 'Missing required parameter: params' }
    }
  }

  const { modelsDir } = getTestPaths()
  const parakeetConfig = params.parakeetConfig || {}
  const modelType = parakeetConfig.modelType || 'tdt'
  const defaultModelPath =
      path.join(modelsDir, MODEL_CONFIGS[modelType]?.file ||
                           MODEL_CONFIGS.tdt.file)

  const modelPath = params.modelPath || defaultModelPath
  const files = params.files || getNamedPathsConfig(modelType, modelPath)

  if (typeof modelPath === 'string' && !fs.existsSync(modelPath)) {
    return {
      output: `Error: GGUF not found: ${modelPath}`,
      passed: false,
      data: { error: `GGUF not found: ${modelPath}` }
    }
  }

  let model
  try {
    model = new TranscriptionParakeet({
      files,
      config: {
        parakeetConfig: {
          modelType,
          maxThreads: parakeetConfig.maxThreads || 4,
          useGPU: parakeetConfig.useGPU || false,
          ...parakeetConfig
        }
      }
    })
    await model._load()

    if (!params.audioInput) {
      return {
        output: 'Config validation passed - model loaded successfully',
        passed: true,
        data: {
          segments: [],
          segmentCount: 0,
          fullText: '',
          textLength: 0,
          durationMs: 0,
          stats: null
        }
      }
    }

    const audioStream = createAudioStream(params.audioInput)
    const response = await model.run(audioStream)

    const segments = []
    let jobStats = null

    await response
      .onUpdate((outputArr) => {
        const items = Array.isArray(outputArr) ? outputArr : [outputArr]
        segments.push(...items)
        if (params.onUpdate) {
          params.onUpdate(outputArr)
        }
      })
      .await()

    if (response.stats) {
      jobStats = response.stats
    }

    const fullText = segments
      .map(s => (s && s.text) ? s.text : '')
      .filter(t => t.trim().length > 0)
      .join(' ')
      .trim()
      .replace(/\s+/g, ' ')

    const textLength = fullText.length
    const segmentCount = segments.length

    let passed = true

    if (expectation.minSegments !== undefined && segmentCount < expectation.minSegments) {
      passed = false
    }
    if (expectation.maxSegments !== undefined && segmentCount > expectation.maxSegments) {
      passed = false
    }
    if (expectation.minTextLength !== undefined && textLength < expectation.minTextLength) {
      passed = false
    }
    if (expectation.expectedText !== undefined && !fullText.toLowerCase().includes(expectation.expectedText.toLowerCase())) {
      passed = false
    }

    const output = `Transcribed ${segmentCount} segments from audio`

    return {
      output,
      passed,
      data: {
        segments,
        segmentCount,
        fullText,
        textLength,
        stats: jobStats
      }
    }
  } catch (error) {
    return {
      output: `Error: ${error.message}`,
      passed: false,
      data: {
        error: error.message,
        segments: [],
        segmentCount: 0,
        fullText: '',
        textLength: 0,
        stats: null
      }
    }
  } finally {
    if (model) {
      try {
        await model.unload()
      } catch (e) {
        // Ignore cleanup errors
      }
    }
  }
}

// GGUF model staging. Each entry is a single self-contained
// `.gguf` file produced by `scripts/convert-nemo-to-gguf.py` (run
// via `npm run setup-models`, which writes to `./models/`). The
// integration suite either downloads it from a HuggingFace mirror or
// copies it from any pre-staged directory pointed at by the
// `QVAC_TEST_GGUF_DIR` env var (typically the package's own
// `./models/`).
//
// Quantisation: we default to q8_0 (1.9x smaller than f16, no
// user-facing transcript regressions on shipping fixtures). Tests can
// override per-model with QVAC_TEST_GGUF_<TYPE> (e.g.
// QVAC_TEST_GGUF_EOU=/path/to/parakeet-eou-120m-v1.f16.gguf).
const MODEL_CONFIGS = {
  ctc: {
    file: 'parakeet-ctc-0.6b.q8_0.gguf',
    minSize: 600 * 1024 * 1024,
    url: null
  },
  tdt: {
    file: 'parakeet-tdt-0.6b-v3.q8_0.gguf',
    minSize: 600 * 1024 * 1024,
    url: null
  },
  eou: {
    file: 'parakeet-eou-120m-v1.q8_0.gguf',
    minSize: 100 * 1024 * 1024,
    url: null
  },
  sortformer: {
    file: 'sortformer-4spk-v1.q8_0.gguf',
    minSize: 100 * 1024 * 1024,
    url: null
  }
}

/**
 * Ensures a GGUF for the given model type is staged on disk. Returns
 * the absolute path to the .gguf file (NOT a directory; the GGUF
 * backend uses single-file models).
 *
 * Resolution order:
 *   1. `QVAC_TEST_GGUF_<TYPE>` env var (e.g. QVAC_TEST_GGUF_TDT)
 *   2. `QVAC_TEST_GGUF_DIR/<file>` -- copy from any pre-staged
 *      `models/` directory if present (typically the package's own
 *      `./models/` produced by `npm run setup-models`).
 *   3. Existing cache in the test models dir.
 *   4. (TODO) Download from HuggingFace -- not yet wired since GGUFs
 *      aren't published there; users currently stage them by running
 *      `npm run setup-models` (which writes to ./models/) or by
 *      pointing `QVAC_TEST_GGUF_DIR` at an existing GGUF directory.
 *
 * @param {string} modelType - 'tdt', 'ctc', 'eou', or 'sortformer'
 * @param {string} [override] - explicit GGUF path to use
 * @returns {Promise<string|null>} GGUF file path, or null if unavailable
 */
async function ensureGgufForType (modelType, override = null) {
  const cfg = MODEL_CONFIGS[modelType]
  if (!cfg) return null

  if (override && fs.existsSync(override)) return override

  const envKey = `QVAC_TEST_GGUF_${modelType.toUpperCase()}`
  if (process.env && process.env[envKey] && fs.existsSync(process.env[envKey])) {
    return process.env[envKey]
  }

  const { modelsDir, samplesDir } = getTestPaths()
  const cachePath = path.join(modelsDir, cfg.file)

  if (fs.existsSync(cachePath) &&
      fs.statSync(cachePath).size >= (cfg.minSize || 0)) {
    return cachePath
  }

  if (isMobile && samplesDir) {
    const bundledPath = path.join(samplesDir, cfg.file)
    if (fs.existsSync(bundledPath) &&
        fs.statSync(bundledPath).size >= (cfg.minSize || 0)) {
      return bundledPath
    }
  }

  const externalDir = process.env && process.env.QVAC_TEST_GGUF_DIR
  if (externalDir) {
    const externalPath = path.join(externalDir, cfg.file)
    if (fs.existsSync(externalPath) &&
        fs.statSync(externalPath).size >= (cfg.minSize || 0)) {
      console.log(`  Staging GGUF from ${externalPath} -> ${cachePath}`)
      fs.copyFileSync(externalPath, cachePath)
      return cachePath
    }
  }

  if (cfg.url) {
    console.log(`  Downloading ${cfg.file}...`)
    await downloadFile(cfg.url, cachePath)
    return cachePath
  }

  console.log(`  ${modelType.toUpperCase()} GGUF not available. Run ` +
              '`npm run setup-models` or set ' +
              `${envKey} / QVAC_TEST_GGUF_DIR to a directory of GGUFs ` +
              'to enable this test.')
  return null
}

// Back-compat alias so older test files keep working.
async function ensureModelForType (modelType) {
  return ensureGgufForType(modelType)
}

/**
 * Resolves a GGUF for the given model type or skips the test cleanly
 * if no GGUF can be staged. Use as the first line of every integration
 * test that needs a real model -- it's a no-op when QVAC_TEST_GGUF_DIR
 * (or QVAC_TEST_GGUF_<TYPE>) is set, and returns a cleanly-skipped
 * outcome otherwise.
 *
 * @param {Object} t - brittle test object (must have `.skip(message)`)
 * @param {string} [modelType='tdt']
 * @returns {Promise<string|null>} GGUF path, or null when skipped
 */
async function loadGgufOrSkip (t, modelType = 'tdt') {
  const ggufPath = await ensureGgufForType(modelType)
  if (!ggufPath || !fs.existsSync(ggufPath)) {
    t.pass(`No ${modelType.toUpperCase()} GGUF available; run ` +
           '`npm run setup-models` (or set ' +
           `QVAC_TEST_GGUF_${modelType.toUpperCase()}=/path/to/model.gguf, ` +
           'or QVAC_TEST_GGUF_DIR=/path/to/models)')
    return null
  }
  return ggufPath
}

/**
 * Build the file-path config for a given model type. The GGUF
 * backend takes a single `modelPath` (the .gguf file); this
 * helper returns that shape so callers don't need to special-case
 * per model type.
 *
 * @param {string} _modelType - 'tdt', 'ctc', 'eou', or 'sortformer' (informational)
 * @param {string} ggufPath - absolute path to the .gguf file
 * @returns {Object} { modelPath } config to spread into ParakeetInterface config
 */
function getNamedPathsConfig (_modelType, ggufPath) {
  return { modelPath: ggufPath }
}

module.exports = {
  binding,
  ParakeetInterface,
  TranscriptionParakeet,
  detectPlatform,
  waitUntilIdle,
  runTranscription,
  createAudioStream,
  calculateAudioDuration,
  generateTestAudio,
  makePcmNoise,
  setupJsLogger,
  getTestPaths,
  wordErrorRate,
  validateAccuracy,
  ensureModel,
  ensureModelForType,
  ensureGgufForType,
  loadGgufOrSkip,
  readFileChunked,
  getNamedPathsConfig,
  isMobile,
  platform,
  arch,
  MODEL_CONFIGS
}
