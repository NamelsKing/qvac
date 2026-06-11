'use strict'

const test = require('brittle')
const os = require('bare-os')
const process = require('bare-process')
const { isMobile, platform, getImagePath, ensureModelPath, runOcrComparison } = require('./utils')

const DESKTOP_TIMEOUT = 120 * 1000 // 2 minutes for desktop

// Guards the opt-in 1x1-conv -> ggml_mul_mat path added in QVAC-20532. The
// default suite runs every conv through ggml_conv_2d (im2col); this test forces
// OCR_GGML_CONV1X1_MULMAT=1 so the CRAFT detector's 1x1 convs take the matmul
// path, then asserts the full EasyOCR pipeline still produces correct output
// (the rewrite must be numerically equivalent to conv_2d for 1x1/stride-1).
//
// The toggle is read by the native addon via getenv at first use, so we set it
// through bare-os (which maps to setenv) before constructing the addon and
// restore it afterwards. Desktop POSIX only: mobile device-farm runs don't
// propagate these process env vars, and on Windows uv_os_setenv
// (SetEnvironmentVariableW) does not update the CRT table that the addon's
// std::getenv reads — so the toggle wouldn't take effect and the addon would
// fall back to the default conv_2d path, giving no coverage of the matmul path.
// The lever is therefore CI-verified on Linux/macOS only.
const ENV_KEYS = ['OCR_GGML_CONV1X1_MULMAT']

test('EasyOCR 1x1-conv mul_mat path (CRAFT)', { timeout: DESKTOP_TIMEOUT }, async function (t) {
  if (isMobile) {
    t.pass('skipped on mobile (env toggle is a desktop-only A/B lever)')
    return
  }
  if (platform === 'win32') {
    t.pass('skipped on win32 (SetEnvironmentVariableW does not reach the addon CRT getenv)')
    return
  }

  const hasGetEnv = typeof os.getEnv === 'function'
  const hasSetEnv = typeof os.setEnv === 'function'
  const prev = new Map()
  for (const key of ENV_KEYS) {
    prev.set(key, (hasGetEnv ? os.getEnv(key) : process.env[key]) || '')
  }

  function setEnv (key, val) {
    if (hasSetEnv) os.setEnv(key, val)
    process.env[key] = val
  }

  function restoreEnv () {
    for (const key of ENV_KEYS) {
      const original = prev.get(key)
      if (original) {
        setEnv(key, original)
        continue
      }
      if (typeof os.unsetEnv === 'function') os.unsetEnv(key)
      else if (hasSetEnv) os.setEnv(key, '')
      // bare-process's env proxy rejects `delete` (TypeError under strict mode); '' is sufficient since the addon reads via std::getenv.
      process.env[key] = ''
    }
  }

  for (const key of ENV_KEYS) setEnv(key, '1')
  try {
    const detectorPath = await ensureModelPath('detector_craft')
    const recognizerPath = await ensureModelPath('recognizer_latin')
    const imagePath = getImagePath('/test/images/basic_test.bmp')

    t.comment('Forcing 1x1-conv mul_mat path; image: ' + imagePath + ', platform: ' + platform)

    await runOcrComparison(t, {
      params: {
        pathDetector: detectorPath,
        pathRecognizer: recognizerPath,
        langList: ['en']
      },
      imagePath,
      runOptions: { paragraph: false },
      perfLabel: '[EasyOCR basic_test 1x1-mulmat]',
      perfOpts: { skipReport: true },
      assertResult (output) {
        t.ok(Array.isArray(output), 'output should be an array')
        t.ok(output.length === 3, `output length should be 3, got ${output.length}`)
        const texts = output.map(o => o[1])
        t.ok(texts.includes('tilted'), 'should contain "tilted"')
        t.ok(texts.includes('normal'), 'should contain "normal"')
        t.ok(texts.includes('vertical'), 'should contain "vertical"')
      }
    })

    t.pass('1x1-conv mul_mat path produced correct OCR output')
  } finally {
    restoreEnv()
  }
})
