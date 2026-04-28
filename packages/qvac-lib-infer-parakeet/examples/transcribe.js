'use strict'

/**
 * Universal transcribe / diarize example.
 *
 * Loads a single Parakeet GGUF (CTC, TDT, EOU, or Sortformer) and
 * runs inference on a wav / raw PCM file. The binding auto-detects
 * the model type from the GGUF metadata, so one example covers every
 * checkpoint.
 *
 * Usage:
 *   bare examples/transcribe.js --model <gguf> --audio <file>
 */

/* global Bare */
const path = require('bare-path')
const process = require('bare-process')
const binding = require('../binding.js')
const { ParakeetInterface } = require('../parakeet.js')
const {
  setupLogger,
  parseWavFile,
  convertRawToFloat32,
  loadModelWeights,
  validatePaths,
  createJobTracker,
  createOutputCallback,
  printResults,
  readFileAsStream
} = require('./utils.js')

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

async function loadAudio (audioPath) {
  const ext = path.extname(audioPath).toLowerCase()
  if (ext === '.wav') return parseWavFile(audioPath)
  const rawBuffer = await readFileAsStream(audioPath)
  return convertRawToFloat32(rawBuffer)
}

async function main () {
  const args = parseArgs()
  if (!args.model || !args.audio) {
    console.error('Usage: bare examples/transcribe.js --model <gguf> --audio <file>')
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

  const audioData = await loadAudio(audioPath)
  const durationS = audioData.length / 16000
  console.log(`Audio: ${durationS.toFixed(2)}s\n`)

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
