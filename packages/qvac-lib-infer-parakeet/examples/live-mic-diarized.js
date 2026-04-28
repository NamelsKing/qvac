'use strict'

/**
 * Live-mic transcription + diarization example.
 *
 * Captures the default input device via `sox -d` and feeds chunks
 * into two streaming binding instances: a Sortformer
 * SortformerStreamSession (stable speaker IDs across chunks via the
 * session's internal rolling history) and an ASR StreamSession.
 * Each printed line is tagged with the dominant speaker for that
 * chunk. Press Ctrl-C to flush and exit.
 *
 * Usage:
 *   bare examples/live-mic-diarized.js \
 *        --asr-model <gguf> --diar-model <gguf> [--accumulate]
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
const HISTORY_MS = 30000
const SAMPLES_PER_CHUNK = Math.floor(SAMPLE_RATE * CHUNK_MS / 1000)
const CAPTURE_CMD = 'sox -d -t raw -r 16000 -b 16 -c 1 -e signed-integer -L -'

const SILENCE_SENTINELS = new Set([
  '[No speech detected]',
  '[Audio too short]',
  '[Model not ready]',
  '[No speakers detected]'
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

function dominantSpeaker (sortformerText, fallback = -1) {
  const totals = new Map()
  for (const line of sortformerText.split('\n')) {
    const m = line.match(/Speaker\s+(\d+)\s*:\s*([\d.:]+)\s*-\s*([\d.:]+)/)
    if (!m) continue
    const toSec = (ts) => {
      const parts = ts.split(':').map(parseFloat)
      if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2]
      if (parts.length === 2) return parts[0] * 60 + parts[1]
      return parts[0]
    }
    const dur = Math.max(0, toSec(m[3]) - toSec(m[2]))
    const id = parseInt(m[1], 10)
    totals.set(id, (totals.get(id) || 0) + dur)
  }
  let bestId = fallback
  let bestDur = 0
  for (const [id, dur] of totals) {
    if (dur > bestDur) { bestDur = dur; bestId = id }
  }
  return bestId
}

function parseArgs () {
  const args = { asrModel: null, diarModel: null, accumulate: false }
  const argv = Bare.argv.slice(2)
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--asr-model' || a === '-m') args.asrModel = argv[++i]
    else if (a === '--diar-model' || a === '-d') args.diarModel = argv[++i]
    else if (a === '--accumulate') args.accumulate = true
  }
  return args
}

function makeBindingCallback (trackerBox, stoppingRef) {
  return (handle, event, jobId, output, error) => {
    const tracker = trackerBox.current
    if (error) {
      if (!stoppingRef.value) console.error('Error:', error)
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
  }
}

async function main () {
  const args = parseArgs()
  if (!args.asrModel || !args.diarModel) {
    console.error('Usage: bare examples/live-mic-diarized.js --asr-model <gguf> --diar-model <gguf> [--accumulate]')
    process.exit(1)
  }

  setupLogger(binding)
  const stopping = { value: false }

  const asrPath = path.resolve(args.asrModel)
  const diarPath = path.resolve(args.diarModel)
  if (!validatePaths({ model: asrPath })) { binding.releaseLogger(); process.exit(1) }
  if (!validatePaths({ model: diarPath })) { binding.releaseLogger(); process.exit(1) }

  console.log(`Loading ${asrPath}...`)
  console.log(`Loading ${diarPath}...`)

  const asrTracker = { current: createJobTracker() }
  const diarTracker = { current: createJobTracker() }

  const asr = new ParakeetInterface(binding, {
    modelPath: asrPath,
    streaming: true,
    streamingChunkMs: CHUNK_MS
  }, makeBindingCallback(asrTracker, stopping), () => {})

  const diar = new ParakeetInterface(binding, {
    modelPath: diarPath,
    streaming: true,
    streamingChunkMs: CHUNK_MS,
    streamingHistoryMs: HISTORY_MS
  }, makeBindingCallback(diarTracker, stopping), () => {})

  await loadModelWeights(asr, asrPath)
  await loadModelWeights(diar, diarPath)
  await asr.activate()
  await diar.activate()
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
  let lineSpeaker = null
  function flushLine () {
    if (lineOpen) { process.stdout.write('\n'); lineOpen = false; lineSpeaker = null }
  }
  function emitTranscript (speaker, text) {
    if (isSilenceText(text)) {
      if (args.accumulate) flushLine()
      return
    }
    const tag = speaker >= 0 ? `speaker_${speaker}` : 'speaker_?'
    const ts = new Date().toISOString().slice(11, 19)
    if (args.accumulate) {
      if (lineOpen && lineSpeaker !== speaker) flushLine()
      if (!lineOpen) {
        process.stdout.write(`[${ts}] ${tag}: ${text}`)
        lineOpen = true
        lineSpeaker = speaker
      } else {
        process.stdout.write(' ' + text)
      }
    } else {
      console.log(`[${ts}] ${tag}: ${text}`)
    }
  }

  let pcmTail = Buffer.alloc(0)
  let floatBuf = new Float32Array(0)
  let processing = Promise.resolve()

  // Sortformer's streaming session emits a segment only when a speaker
  // span boundary is crossed (start / stop / change). While the same
  // speaker keeps talking across multiple chunks, no new segment
  // fires, so the per-chunk sortformer output for those chunks is
  // empty and dominantSpeaker() would return -1 ("speaker_?"). Cache
  // the last speaker we saw and reuse it whenever this chunk has ASR
  // text but no fresh sortformer segment -- the speaker hasn't
  // changed, only Sortformer hasn't repeated itself.
  let lastSpeaker = -1

  function emit (chunk) {
    processing = processing
      .then(async () => {
        asrTracker.current = createJobTracker()
        diarTracker.current = createJobTracker()
        const at = asrTracker.current
        const dt = diarTracker.current

        const asrPromise = (async () => {
          await asr.append({ type: 'audio', data: chunk.buffer })
          await asr.append({ type: 'end of job' })
          await Promise.race([at.promise, new Promise(resolve => setTimeout(resolve, 30000))])
        })()

        const diarPromise = (async () => {
          await diar.append({ type: 'audio', data: chunk.buffer })
          await diar.append({ type: 'end of job' })
          await Promise.race([dt.promise, new Promise(resolve => setTimeout(resolve, 30000))])
        })()

        await Promise.all([asrPromise, diarPromise])

        const text = at.transcriptions
          .map(s => s.text).join(' ').trim().replace(/\s+/g, ' ')
        const sortformerText = dt.transcriptions.map(s => s.text).join('\n')
        let speaker = dominantSpeaker(sortformerText, -1)
        if (speaker >= 0) {
          lastSpeaker = speaker
        } else if (text.length > 0 && lastSpeaker >= 0) {
          // Same speaker continuing -- Sortformer didn't repeat the
          // segment, but ASR proves someone is still talking.
          speaker = lastSpeaker
        }
        emitTranscript(speaker, text)
      })
      .catch(err => {
        if (!stopping.value) console.error('Inference error:', err.message)
      })
  }

  child.stdout.on('data', (chunk) => {
    if (stopping.value) return
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
    if (stopping.value) return
    stopping.value = true
    console.log('\nStopping...')
    try { child.kill('SIGTERM') } catch (e) {}
    if (floatBuf.length > 0) emit(floatBuf)
    await processing
    flushLine()
    try { await asr.destroyInstance() } catch (e) {}
    try { await diar.destroyInstance() } catch (e) {}
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
