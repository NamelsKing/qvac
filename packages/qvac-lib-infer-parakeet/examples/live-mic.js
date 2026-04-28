'use strict'

/**
 * Live-mic transcription example.
 *
 * Captures the default input device via `sox -d` (16 kHz mono s16le),
 * pushes each chunk through a pushable async-iterable, and feeds it
 * to the public `TranscriptionParakeet` class via `model.run()`.
 * Press Ctrl-C to flush and exit.
 *
 * Usage:
 *   bare examples/live-mic.js --model <gguf> [--accumulate]
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
  '[Model not ready]'
])

function isSilenceText (text) {
  return text.length === 0 || SILENCE_SENTINELS.has(text)
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

  setupLogger(addonLogging)
  let stopping = false

  const modelPath = path.resolve(args.model)
  if (!validatePaths({ model: modelPath })) {
    addonLogging.releaseLogger()
    process.exit(1)
  }

  console.log(`Loading ${modelPath}...`)

  const model = new TranscriptionParakeet({
    files: { model: modelPath },
    config: {
      parakeetConfig: {
        streaming: true,
        streamingChunkMs: 2000
      }
    }
  })
  await model.load()
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

  // ---------- Live segment printing ----------
  let lineOpen = false
  function flushLine () {
    if (lineOpen) {
      process.stdout.write('\n')
      lineOpen = false
    }
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

  // ---------- Audio fan-in ----------
  const audioStream = pushableStream()
  child.stdout.on('data', (chunk) => {
    if (!stopping) audioStream.push(chunk)
  })

  // ---------- Run inference ----------
  const runPromise = (async () => {
    const response = await model.run(audioStream)
    await response
      .onUpdate(out => {
        const items = Array.isArray(out) ? out : [out]
        // Each segment from the streaming session is a self-contained
        // transcript chunk. Collapse to a single line per emit.
        const text = items
          .filter(s => s && s.text && s.toAppend)
          .map(s => s.text)
          .join(' ')
          .trim()
          .replace(/\s+/g, ' ')
        emitTranscript(text)
      })
      .await()
  })()

  // ---------- Shutdown ----------
  async function shutdown () {
    if (stopping) return
    stopping = true
    console.log('\nStopping...')
    try { child.kill('SIGTERM') } catch (e) { /* ignore */ }
    audioStream.end()
    try { await runPromise } catch (e) { /* swallow */ }
    flushLine()
    try { await model.unload() } catch (e) { /* ignore */ }
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
