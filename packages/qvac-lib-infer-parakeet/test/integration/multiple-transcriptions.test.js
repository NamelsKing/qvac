'use strict'

const test = require('brittle')
const path = require('bare-path')
const fs = require('bare-fs')
const {
  binding,
  TranscriptionParakeet,
  detectPlatform,
  setupJsLogger,
  getTestPaths,
  loadGgufOrSkip,
  isMobile
} = require('./helpers.js')

const platform = detectPlatform()
const { modelPath, samplesDir } = getTestPaths()

function loadAudio (samplePath) {
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
      for (const seg of items) {
        if (seg && seg.text) segments.push(seg)
      }
    })
    .await()
  return segments
}

/**
 * Test that multiple consecutive transcriptions on the same model
 * instance work without errors.
 */
test('Multiple consecutive transcriptions should work without errors', { timeout: 600000 }, async (t) => {
  const NUM_TRANSCRIPTIONS = 3
  const loggerBinding = setupJsLogger(binding)

  console.log('\n' + '='.repeat(60))
  console.log('MULTIPLE CONSECUTIVE TRANSCRIPTIONS TEST')
  console.log('='.repeat(60))
  console.log(` Platform: ${platform}`)
  console.log(` Model path: ${modelPath}`)
  console.log(` Number of transcriptions: ${NUM_TRANSCRIPTIONS}`)
  console.log(` Mobile: ${isMobile}`)
  console.log('='.repeat(60) + '\n')

  const stagedGguf = await loadGgufOrSkip(t)
  if (!stagedGguf) return

  const samplePath = path.join(samplesDir, 'sample.raw')
  if (!fs.existsSync(samplePath)) {
    loggerBinding.releaseLogger()
    t.pass('Test skipped - sample audio not found')
    return
  }

  const audioData = loadAudio(samplePath)
  console.log(`   Audio duration: ${(audioData.length / 16000).toFixed(2)}s\n`)

  const model = new TranscriptionParakeet({
    files: { model: stagedGguf },
    config: { parakeetConfig: { maxThreads: 4, useGPU: false } }
  })

  const allResults = []
  const timings = []

  try {
    await model.load()

    for (let run = 1; run <= NUM_TRANSCRIPTIONS; run++) {
      console.log(`=== Transcription ${run}/${NUM_TRANSCRIPTIONS} ===`)
      const runStartTime = Date.now()

      const segments = await transcribe(model, audioData)
      const runTime = Date.now() - runStartTime
      timings.push(runTime)

      const runText = segments.map(s => s.text).join(' ').trim()
      allResults.push({ run, segments, text: runText })

      console.log(`   Time: ${runTime}ms`)
      console.log(`   Segments: ${segments.length}`)
      console.log(`   Text preview: "${runText.substring(0, 80)}${runText.length > 80 ? '...' : ''}"\n`)

      if (run < NUM_TRANSCRIPTIONS) {
        await new Promise(resolve => setTimeout(resolve, 200))
      }
    }

    console.log('='.repeat(60))
    console.log('TEST SUMMARY')
    console.log('='.repeat(60))
    timings.forEach((time, i) => console.log(`    Run ${i + 1}: ${time}ms`))
    const avgTime = timings.reduce((a, b) => a + b, 0) / timings.length
    console.log(`\n  Average time: ${avgTime.toFixed(0)}ms`)
    const totalSegments = allResults.reduce((acc, r) => acc + r.segments.length, 0)
    console.log(`  Total segments: ${totalSegments}`)
    console.log('='.repeat(60) + '\n')

    t.ok(timings.length === NUM_TRANSCRIPTIONS,
      `Should complete ${NUM_TRANSCRIPTIONS} transcriptions (got ${timings.length})`)
    t.ok(allResults.every(r => r.segments.length > 0),
      'Every run should produce segments')
  } finally {
    try { await model.unload() } catch (e) { /* ignore */ }
    try { loggerBinding.releaseLogger() } catch (e) { /* ignore */ }
  }
})

/**
 * Test that creating fresh model instances for each transcription
 * works correctly. Simulates app-restart scenarios.
 */
test('Fresh model instance per transcription (app restart simulation)', { timeout: 600000 }, async (t) => {
  const NUM_INSTANCES = 2
  const loggerBinding = setupJsLogger(binding)

  console.log('\n' + '='.repeat(60))
  console.log('FRESH INSTANCE PER TRANSCRIPTION TEST')
  console.log('='.repeat(60))
  console.log(` Platform: ${platform}`)
  console.log(` Instances to create: ${NUM_INSTANCES}`)
  console.log('='.repeat(60) + '\n')

  const stagedGguf = await loadGgufOrSkip(t)
  if (!stagedGguf) return

  const samplePath = path.join(samplesDir, 'sample.raw')
  if (!fs.existsSync(samplePath)) {
    loggerBinding.releaseLogger()
    t.pass('Test skipped - sample audio not found')
    return
  }

  const audioData = loadAudio(samplePath)
  const results = []

  for (let instance = 1; instance <= NUM_INSTANCES; instance++) {
    console.log(`--- Instance ${instance}/${NUM_INSTANCES} ---`)
    const instanceStartTime = Date.now()

    const model = new TranscriptionParakeet({
      files: { model: stagedGguf },
      config: { parakeetConfig: { maxThreads: 4, useGPU: false } }
    })
    try {
      await model.load()
      const loadTime = Date.now() - instanceStartTime

      const segments = await transcribe(model, audioData)
      const totalTime = Date.now() - instanceStartTime
      const transcriptionTime = totalTime - loadTime
      const fullText = segments.map(s => s.text).join(' ').trim()

      console.log(`   Load time: ${loadTime}ms`)
      console.log(`   Transcription time: ${transcriptionTime}ms`)
      console.log(`   Total time: ${totalTime}ms`)
      console.log(`   Segments: ${segments.length}\n`)

      results.push({
        loadTime,
        transcriptionTime,
        totalTime,
        segmentCount: segments.length,
        textLength: fullText.length
      })
    } finally {
      try { await model.unload() } catch (e) { /* ignore */ }
    }

    if (instance < NUM_INSTANCES) {
      await new Promise(resolve => setTimeout(resolve, 500))
    }
  }

  console.log('='.repeat(60))
  console.log('FRESH INSTANCE SUMMARY')
  console.log('='.repeat(60))
  results.forEach((r, i) => {
    console.log(`  Instance ${i + 1}:`)
    console.log(`    Load: ${r.loadTime}ms`)
    console.log(`    Transcribe: ${r.transcriptionTime}ms`)
    console.log(`    Total: ${r.totalTime}ms`)
    console.log(`    Segments: ${r.segmentCount}`)
  })
  console.log('='.repeat(60) + '\n')

  t.ok(results.length === NUM_INSTANCES, `Created ${NUM_INSTANCES} fresh model instances`)
  t.ok(results.every(r => r.segmentCount > 0), 'All instances should produce segments')

  try { loggerBinding.releaseLogger() } catch (e) { /* ignore */ }
})
