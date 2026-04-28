'use strict'

/**
 * Live-mic transcription example.
 *
 * Captures the default input device via `sox -d` (16 kHz mono s16le)
 * and streams chunks into the binding's streaming session. Press
 * Ctrl-C to flush and exit.
 *
 * Usage:
 *   bare examples/live-mic.js --model <gguf> [--accumulate]
 */

/* global Bare */
const path = require('bare-path')
const process = require('bare-process')
const subprocess = require('bare-subprocess')
const binding = require('../binding.js')
const { ParakeetInterface } = require('../parakeet.js')
const {
  setupLogger,
  loadModelWeights,
  validatePaths,
  createJobTracker
} = require('./utils.js')

const SAMPLE_RATE = 16000
const CHUNK_MS = 2000
const SAMPLES_PER_CHUNK = Math.floor(SAMPLE_RATE * CHUNK_MS / 1000)
const CAPTURE_CMD = 'sox -d -t raw -r 16000 -b 16 -c 1 -e signed-integer -L -'

const SILENCE_SENTINELS = new Set([
  '[No speech detected]',
  '[Audio too short]',
  '[Model not ready]'
])

function isSilenceText (text) {
  return text.length === 0 || SILENCE_SENTINELS.has(text)
}

function pcmInt16ToFloat32 (buf) {
  const evenBytes = buf.length - (buf.length % 2)
  const view = new DataView(buf.buffer, buf.byteOffset, evenBytes)
  const samples = new Float32Array(evenBytes / 2)
  for (let i = 0; i < samples.length; i++) {
    samples[i] = view.getInt16(i * 2, true) / 32768
  }
  return { samples, consumed: evenBytes }
}

function parseArgs () {
  const args = { model: null, accumulate: false }
  const argv = Bare.argv.slice(2)
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--model' || a === '-m') args.model = argv[++i]
    else if (a === '--accumulate') args.accumulate = true
  }
  return args
}

async function main () {
  const args = parseArgs()
  if (!args.model) {
    console.error('Usage: bare examples/live-mic.js --model <gguf> [--accumulate]')
    process.exit(1)
  }

  setupLogger(binding)
  let stopping = false

  const modelPath = path.resolve(args.model)
  if (!validatePaths({ model: modelPath })) {
    binding.releaseLogger()
    process.exit(1)
  }

  console.log(`Loading ${modelPath}...`)

  const activeTracker = { current: createJobTracker() }
  const parakeet = new ParakeetInterface(binding, {
    modelPath,
    streaming: true,
    streamingChunkMs: CHUNK_MS
  }, (handle, event, jobId, output, error) => {
    const tracker = activeTracker.current
    if (error) {
      if (!stopping) console.error('Error:', error)
      return
    }
    if (event === 'Output' && output) {
      const segs = Array.isArray(output) ? output : [output]
      for (const s of segs) {
        if (s && s.text && s.toAppend) tracker.transcriptions.push(s)
      }
      tracker.markOutput()
    }
    if (event === 'JobEnded') tracker.markJobEnded()
  }, () => {})

  await loadModelWeights(parakeet, modelPath)
  await parakeet.activate()
  console.log('Listening (Ctrl-C to stop)...\n')

  const [bin, ...rest] = CAPTURE_CMD.split(/\s+/)
  const child = subprocess.spawn(bin, rest, { stdio: ['ignore', 'pipe', 'pipe'] })
  child.on('error', (err) => {
    console.error(`\nFailed to spawn "${bin}": ${err.message}`)
    console.error('Install sox (brew install sox / apt install sox / choco install sox).')
    process.exit(1)
  })
  child.stderr.on('data', () => {})

  let lineOpen = false
  function flushLine () {
    if (lineOpen) { process.stdout.write('\n'); lineOpen = false }
  }
  function emitTranscript (text) {
    if (isSilenceText(text)) {
      if (args.accumulate) flushLine()
      return
    }
    const ts = new Date().toISOString().slice(11, 19)
    if (args.accumulate) {
      if (!lineOpen) {
        process.stdout.write(`[${ts}] ${text}`)
        lineOpen = true
      } else {
        process.stdout.write(' ' + text)
      }
    } else {
      console.log(`[${ts}] ${text}`)
    }
  }

  let pcmTail = Buffer.alloc(0)
  let floatBuf = new Float32Array(0)
  let processing = Promise.resolve()

  function emit (chunk) {
    processing = processing
      .then(() => {
        activeTracker.current = createJobTracker()
        const tracker = activeTracker.current
        return parakeet.append({ type: 'audio', data: chunk.buffer })
          .then(() => parakeet.append({ type: 'end of job' }))
          .then(() => Promise.race([
            tracker.promise,
            new Promise(resolve => setTimeout(resolve, 30000))
          ]))
          .then(() => {
            const text = tracker.transcriptions
              .map(s => s.text).join(' ').trim().replace(/\s+/g, ' ')
            emitTranscript(text)
          })
      })
      .catch(err => {
        if (!stopping) console.error('Inference error:', err.message)
      })
  }

  child.stdout.on('data', (chunk) => {
    if (stopping) return
    const merged = Buffer.concat([pcmTail, chunk])
    const { samples, consumed } = pcmInt16ToFloat32(merged)
    pcmTail = merged.slice(consumed)
    if (samples.length === 0) return
    const next = new Float32Array(floatBuf.length + samples.length)
    next.set(floatBuf, 0)
    next.set(samples, floatBuf.length)
    floatBuf = next
    while (floatBuf.length >= SAMPLES_PER_CHUNK) {
      const out = floatBuf.slice(0, SAMPLES_PER_CHUNK)
      floatBuf = floatBuf.slice(SAMPLES_PER_CHUNK)
      emit(out)
    }
  })

  async function shutdown () {
    if (stopping) return
    stopping = true
    console.log('\nStopping...')
    try { child.kill('SIGTERM') } catch (e) {}
    if (floatBuf.length > 0) emit(floatBuf)
    await processing
    flushLine()
    try { await parakeet.destroyInstance() } catch (e) {}
    binding.releaseLogger()
    process.exit(0)
  }

  process.once('SIGINT', shutdown)
  process.once('SIGTERM', shutdown)
  child.on('exit', () => shutdown())
}

main().catch(err => {
  console.error('Error:', err)
  binding.releaseLogger()
  process.exit(1)
})
