'use strict'

const fs = require('bare-fs')
const path = require('bare-path')
const test = require('brittle')
const {
  binding,
  TranscriptionParakeet,
  detectPlatform,
  setupJsLogger,
  getTestPaths,
  validateAccuracy,
  ensureModel
} = require('./helpers.js')

const platform = detectPlatform()
const { modelPath, samplesDir } = getTestPaths()

function loadRawAudio (samplePath) {
  const rawBuffer = fs.readFileSync(samplePath)
  const pcmData = new Int16Array(rawBuffer.buffer, rawBuffer.byteOffset, rawBuffer.length / 2)
  const audioData = new Float32Array(pcmData.length)
  for (let i = 0; i < pcmData.length; i++) audioData[i] = pcmData[i] / 32768.0
  return audioData
}

async function transcribe (model, audio) {
  const segments = []
  const response = await model.run(audio)
  await response
    .onUpdate(out => {
      const items = Array.isArray(out) ? out : [out]
      for (const s of items) {
        if (s && s.text && s.toAppend) segments.push(s)
      }
    })
    .await()
  return segments
}

test('English transcription and WER verification', { timeout: 300000 }, async (t) => {
  const loggerBinding = setupJsLogger(binding)

  console.log('\n' + '='.repeat(60))
  console.log('PARAKEET TRANSCRIPTION TEST')
  console.log('='.repeat(60))
  console.log(` Platform: ${platform}`)
  console.log(` Model path: ${modelPath}`)

  const stagedGguf = await ensureModel(modelPath)
  if (!stagedGguf || !fs.existsSync(stagedGguf)) {
    t.pass('No GGUF available; set QVAC_TEST_GGUF_DIR=~/dev/qvac-parakeet.cpp/models')
    return
  }
  t.ok(fs.existsSync(stagedGguf), `GGUF exists at ${stagedGguf}`)

  const samplePath = path.join(samplesDir, 'sample.raw')
  if (!fs.existsSync(samplePath)) {
    loggerBinding.releaseLogger()
    t.fail(`Sample audio not found: ${samplePath}`)
    return
  }

  const expectedText = 'Alice was beginning to get very tired of sitting by her sister on the bank and of having nothing to do. Once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it. And what is the use of a book thought Alice without pictures or conversations'

  const model = new TranscriptionParakeet({
    files: { model: stagedGguf },
    config: { parakeetConfig: { maxThreads: 4, useGPU: false } }
  })

  try {
    await model.load()

    const audioData = loadRawAudio(samplePath)
    console.log(`   Audio duration: ${(audioData.length / 16000).toFixed(2)}s`)

    const segments = await transcribe(model, audioData)
    const fullText = segments.map(s => s.text).join(' ').trim()

    t.ok(segments.length > 0, `Should produce segments (got ${segments.length})`)
    t.ok(fullText.length > 0, `Should produce text (got ${fullText.length} chars)`)

    console.log('\n=== TRANSCRIPTION OUTPUT ===')
    console.log(fullText)
    console.log('=== END TRANSCRIPTION ===\n')

    const werResult = validateAccuracy(expectedText, fullText, 0.3)
    console.log(`Expected: "${expectedText.substring(0, 100)}..."`)
    console.log(`Got:      "${fullText.substring(0, 100)}..."`)
    console.log(`>>> Word Error Rate: ${werResult.werPercent}`)

    t.ok(werResult.wer <= 0.3, `WER should be <= 30% (got ${werResult.werPercent})`)
  } finally {
    try { await model.unload() } catch (e) { /* ignore */ }
    try { loggerBinding.releaseLogger() } catch (e) { /* ignore */ }
  }
})
