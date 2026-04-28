'use strict'

/**
 * Combined ASR + diarization example.
 *
 * Runs Sortformer to find speaker time-segments, then transcribes
 * each speaker's audio slice with the ASR model. Output is a
 * "Speaker N: ..." per-segment transcript.
 *
 * Usage:
 *   bare examples/diarized-transcribe.js \
 *        --asr-model <gguf> --diar-model <gguf> --audio <file>
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
  readFileAsStream
} = require('./utils.js')

const SAMPLE_RATE = 16000

function parseArgs () {
  const args = { asrModel: null, diarModel: null, audio: null }
  const argv = Bare.argv.slice(2)
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--asr-model' || a === '-m') args.asrModel = argv[++i]
    else if (a === '--diar-model' || a === '-d') args.diarModel = argv[++i]
    else if (a === '--audio' || a === '-a') args.audio = argv[++i]
  }
  return args
}

function parseSpeakerSegments (sortformerText) {
  const segments = []
  for (const line of sortformerText.split('\n')) {
    const m = line.match(/Speaker\s+(\d+)\s*:\s*([\d.:]+)\s*-\s*([\d.:]+)/)
    if (!m) continue
    const toSec = (ts) => {
      const parts = ts.split(':').map(parseFloat)
      if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2]
      if (parts.length === 2) return parts[0] * 60 + parts[1]
      return parts[0]
    }
    segments.push({
      speaker: parseInt(m[1], 10),
      start: toSec(m[2]),
      end: toSec(m[3])
    })
  }
  segments.sort((a, b) => a.start - b.start)
  return segments
}

function sliceAudio (audioData, startS, endS) {
  const i0 = Math.max(0, Math.floor(startS * SAMPLE_RATE))
  const i1 = Math.min(audioData.length, Math.ceil(endS * SAMPLE_RATE))
  if (i1 <= i0) return null
  return audioData.slice(i0, i1)
}

async function loadAudio (audioPath) {
  const ext = path.extname(audioPath).toLowerCase()
  if (ext === '.wav') return parseWavFile(audioPath)
  const rawBuffer = await readFileAsStream(audioPath)
  return convertRawToFloat32(rawBuffer)
}

async function main () {
  const args = parseArgs()
  if (!args.asrModel || !args.diarModel || !args.audio) {
    console.error('Usage: bare examples/diarized-transcribe.js --asr-model <gguf> --diar-model <gguf> --audio <file>')
    process.exit(1)
  }

  setupLogger(binding)
  const asrModel = path.resolve(args.asrModel)
  const diarModel = path.resolve(args.diarModel)
  const audioPath = path.resolve(args.audio)
  if (!validatePaths({ model: asrModel, audio: audioPath })) {
    binding.releaseLogger()
    process.exit(1)
  }
  if (!validatePaths({ model: diarModel })) {
    binding.releaseLogger()
    process.exit(1)
  }

  console.log(`ASR:   ${asrModel}`)
  console.log(`Diar:  ${diarModel}`)
  console.log(`Audio: ${audioPath}\n`)

  const audioData = await loadAudio(audioPath)
  console.log(`Audio: ${(audioData.length / SAMPLE_RATE).toFixed(2)}s\n`)

  const sfTracker = createJobTracker()
  const sf = new ParakeetInterface(binding, { modelPath: diarModel },
    (handle, event, id, output) => {
      if (event === 'Output' && output) {
        const segs = Array.isArray(output) ? output : [output]
        for (const s of segs) if (s && s.text) sfTracker.transcriptions.push(s)
        sfTracker.markOutput()
      }
      if (event === 'JobEnded') sfTracker.markJobEnded()
    })
  await loadModelWeights(sf, diarModel)
  await sf.activate()
  await sf.append({ type: 'audio', data: audioData.buffer })
  await sf.append({ type: 'end of job' })

  const sfTimeout = setTimeout(() => sfTracker.resolve(),
    Math.max(30000, audioData.length / SAMPLE_RATE * 2000))
  await sfTracker.promise
  clearTimeout(sfTimeout)

  const sfText = sfTracker.transcriptions.map(s => s.text).join(' ').trim()
  await sf.destroyInstance()

  const segments = parseSpeakerSegments(sfText)
  if (segments.length === 0) {
    console.log('No speakers detected.')
    binding.releaseLogger()
    return
  }

  const activeTracker = { current: createJobTracker() }
  const asr = new ParakeetInterface(binding, { modelPath: asrModel },
    (handle, event, id, output) => {
      const tracker = activeTracker.current
      if (event === 'Output' && output) {
        const segs = Array.isArray(output) ? output : [output]
        for (const s of segs) if (s && s.text && s.toAppend) tracker.transcriptions.push(s)
        tracker.markOutput()
      }
      if (event === 'JobEnded') tracker.markJobEnded()
    })
  await loadModelWeights(asr, asrModel)
  await asr.activate()

  const results = []
  for (const seg of segments) {
    const slice = sliceAudio(audioData, seg.start, seg.end)
    if (!slice) {
      results.push({ speaker: seg.speaker, text: '[no audio]' })
      continue
    }
    activeTracker.current = createJobTracker()
    const tracker = activeTracker.current
    await asr.append({ type: 'audio', data: slice.buffer })
    await asr.append({ type: 'end of job' })
    const t = setTimeout(() => tracker.resolve(),
      Math.max(30000, (seg.end - seg.start) * 4000))
    await tracker.promise
    clearTimeout(t)
    const text = tracker.transcriptions.map(s => s.text).join(' ').trim()
    results.push({ speaker: seg.speaker, text: text || '[no speech]' })
  }
  await asr.destroyInstance()

  console.log('\n=== Diarized Transcription ===')
  for (const e of results) console.log(`Speaker ${e.speaker}: ${e.text}`)

  binding.releaseLogger()
}

main().catch(err => {
  console.error('Error:', err)
  binding.releaseLogger()
  process.exit(1)
})
