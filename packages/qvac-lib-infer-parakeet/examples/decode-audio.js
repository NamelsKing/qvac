'use strict'

/**
 * Decode + transcribe example.
 *
 * Same flag surface as `examples/transcribe.js`, but pipes the input
 * audio through `@qvac/decoder-audio` (FFmpeg) first so any
 * container/codec FFmpeg supports (mp3, m4a, ogg, flac, mp4, ...)
 * works -- not just 16 kHz mono .wav / raw s16le PCM.
 *
 * Usage:
 *   bare examples/decode-audio.js --model <gguf> --audio <file>
 */

/* global Bare */
const fs = require('bare-fs')
const path = require('bare-path')
const process = require('bare-process')
const { FFmpegDecoder } = require('@qvac/decoder-audio')
const binding = require('../binding.js')
const { ParakeetInterface } = require('../parakeet.js')
const {
  setupLogger,
  loadModelWeights,
  validatePaths,
  createJobTracker,
  createOutputCallback,
  printResults
} = require('./utils.js')

const SAMPLE_RATE = 16000

function parseArgs () {
  const args = { model: null, audio: null }
  const argv = Bare.argv.slice(2)
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--model' || a === '-m') args.model = argv[++i]
    else if (a === '--audio' || a === '-a') args.audio = argv[++i]
  }
  return args
}

// Silent logger -- the decoder itself is an implementation detail
// here; only its output (the decoded PCM) matters.
const silentLogger = {
  debug () {}, info () {}, warn () {}, error () {}
}

async function decodeToFloat32 (audioPath) {
  const decoder = new FFmpegDecoder({
    config: { streamIndex: 0 },
    logger: silentLogger
  })
  await decoder.load()
  try {
    const chunks = []
    let totalSamples = 0
    const audioStream = fs.createReadStream(audioPath)
    const response = await decoder.run(audioStream)

    response.on('output', (data) => {
      // outputArray is a Uint8Array view of 16 kHz mono s16le PCM.
      const view = new DataView(data.outputArray.buffer,
        data.outputArray.byteOffset,
        data.outputArray.byteLength)
      const n = Math.floor(data.outputArray.byteLength / 2)
      const f32 = new Float32Array(n)
      for (let i = 0; i < n; i++) {
        f32[i] = view.getInt16(i * 2, true) / 32768
      }
      chunks.push(f32)
      totalSamples += n
    })

    await new Promise((resolve, reject) => {
      response.on('end', resolve)
      response.on('error', reject)
    })

    const merged = new Float32Array(totalSamples)
    let offset = 0
    for (const c of chunks) {
      merged.set(c, offset)
      offset += c.length
    }
    return merged
  } finally {
    await decoder.unload()
  }
}

async function main () {
  const args = parseArgs()
  if (!args.model || !args.audio) {
    console.error('Usage: bare examples/decode-audio.js --model <gguf> --audio <file>')
    process.exit(1)
  }

  setupLogger(binding)
  const modelPath = path.resolve(args.model)
  const audioPath = path.resolve(args.audio)
  if (!validatePaths({ model: modelPath, audio: audioPath })) {
    binding.releaseLogger()
    process.exit(1)
  }

  console.log(`Model: ${modelPath}`)
  console.log(`Audio: ${audioPath}\n`)

  const tracker = createJobTracker()
  const parakeet = new ParakeetInterface(
    binding,
    { modelPath },
    createOutputCallback(tracker),
    () => {})

  await loadModelWeights(parakeet, modelPath)
  await parakeet.activate()

  const audioData = await decodeToFloat32(audioPath)
  const durationS = audioData.length / SAMPLE_RATE
  console.log(`Audio: ${durationS.toFixed(2)}s (decoded)\n`)

  await parakeet.append({ type: 'audio', data: audioData.buffer })
  await parakeet.append({ type: 'end of job' })

  const timeoutMs = Math.max(30000, durationS * 2000)
  const timeout = setTimeout(() => tracker.resolve(), timeoutMs)
  await tracker.promise
  clearTimeout(timeout)

  printResults(tracker.transcriptions)
  await parakeet.destroyInstance()
  binding.releaseLogger()
}

main().catch(err => {
  console.error('Error:', err)
  binding.releaseLogger()
  process.exit(1)
})
