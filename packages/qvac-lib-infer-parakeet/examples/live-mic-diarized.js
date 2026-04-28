'use strict'

/**
 * Live-mic transcription + diarization example.
 *
 * Captures the default input device via `sox -d`, fans each chunk
 * out to two pushable async-iterables (one for the ASR engine, one
 * for the Sortformer engine), and feeds both to the public
 * `TranscriptionParakeet` class. Each printed line is tagged with
 * the dominant speaker for that chunk; press Ctrl-C to flush and
 * exit.
 *
 * Usage:
 *   bare examples/live-mic-diarized.js \
 *        --asr-model <gguf> --diar-model <gguf> [--accumulate]
 */

/* global Bare */
const path = require('bare-path')
const process = require('bare-process')
const subprocess = require('bare-subprocess')
const TranscriptionParakeet = require('../index.js')
const addonLogging = require('../addonLogging.js')
const { setupLogger, validatePaths, pushableStream } = require('./utils.js')

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

function dominantSpeaker (sortformerSegments, fallback = -1) {
  const totals = new Map()
  for (const seg of sortformerSegments) {
    const m = seg.text.match(/Speaker\s+(\d+)\s*:\s*([\d.:]+)\s*-\s*([\d.:]+)/)
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

async function main () {
  const args = parseArgs()
  if (!args.asrModel || !args.diarModel) {
    console.error('Usage: bare examples/live-mic-diarized.js --asr-model <gguf> --diar-model <gguf> [--accumulate]')
    process.exit(1)
  }

  setupLogger(addonLogging)
  let stopping = false

  const asrPath = path.resolve(args.asrModel)
  const diarPath = path.resolve(args.diarModel)
  if (!validatePaths({ model: asrPath })) { addonLogging.releaseLogger(); process.exit(1) }
  if (!validatePaths({ model: diarPath })) { addonLogging.releaseLogger(); process.exit(1) }

  console.log(`Loading ${asrPath}...`)
  console.log(`Loading ${diarPath}...`)

  const asr = new TranscriptionParakeet({
    files: { model: asrPath },
    config: {
      parakeetConfig: {
        streaming: true,
        streamingChunkMs: 2000
      }
    }
  })
  const diar = new TranscriptionParakeet({
    files: { model: diarPath },
    config: {
      parakeetConfig: {
        streaming: true,
        streamingChunkMs: 2000,
        streamingHistoryMs: 30000
      }
    }
  })

  await asr.load()
  await diar.load()
  console.log('Listening (Ctrl-C to stop)...\n')

  const child = subprocess.spawn(CAPTURE_CMD.split(' ')[0],
    CAPTURE_CMD.split(' ').slice(1),
    { stdio: ['ignore', 'pipe', 'pipe'] })
  child.on('error', (err) => {
    console.error(`\nFailed to spawn capture command: ${err.message}`)
    console.error('Install sox (brew install sox / apt install sox / choco install sox).')
    process.exit(1)
  })
  child.stderr.on('data', () => {})

  // ---------- Output helpers ----------
  let lineOpen = false
  let lineSpeaker = null
  let lastSpeaker = -1

  function flushLine () {
    if (lineOpen) {
      process.stdout.write('\n')
      lineOpen = false
      lineSpeaker = null
    }
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

  // ---------- Audio fan-out ----------
  const asrStream = pushableStream()
  const diarStream = pushableStream()
  child.stdout.on('data', (chunk) => {
    if (stopping) return
    asrStream.push(chunk)
    diarStream.push(chunk)
  })

  // ---------- Diarization side: maintain a rolling list of recent
  // ---------- Sortformer segments so we can resolve `lastSpeaker`.
  const recentDiarSegments = []
  const diarRunPromise = (async () => {
    const response = await diar.run(diarStream)
    await response
      .onUpdate(out => {
        const items = Array.isArray(out) ? out : [out]
        for (const s of items) {
          if (!s || !s.text) continue
          if (isSilenceText(s.text)) continue
          recentDiarSegments.push(s)
          if (recentDiarSegments.length > 64) recentDiarSegments.shift()
          const speaker = dominantSpeaker([s], -1)
          if (speaker >= 0) lastSpeaker = speaker
        }
      })
      .await()
  })()

  // ---------- ASR side: each segment is tagged with `lastSpeaker`,
  // ---------- which the diarization side keeps fresh.
  const asrRunPromise = (async () => {
    const response = await asr.run(asrStream)
    await response
      .onUpdate(out => {
        const items = Array.isArray(out) ? out : [out]
        const text = items
          .filter(s => s && s.text && s.toAppend)
          .map(s => s.text)
          .join(' ')
          .trim()
          .replace(/\s+/g, ' ')
        emitTranscript(lastSpeaker, text)
      })
      .await()
  })()

  // ---------- Shutdown ----------
  async function shutdown () {
    if (stopping) return
    stopping = true
    console.log('\nStopping...')
    try { child.kill('SIGTERM') } catch (e) { /* ignore */ }
    asrStream.end()
    diarStream.end()
    try { await Promise.all([asrRunPromise, diarRunPromise]) } catch (e) { /* swallow */ }
    flushLine()
    try { await asr.unload() } catch (e) { /* ignore */ }
    try { await diar.unload() } catch (e) { /* ignore */ }
    addonLogging.releaseLogger()
    process.exit(0)
  }

  process.once('SIGINT', shutdown)
  process.once('SIGTERM', shutdown)
  child.on('exit', () => shutdown())
}

main().catch(err => {
  console.error('Error:', err)
  addonLogging.releaseLogger()
  process.exit(1)
})
