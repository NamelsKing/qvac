'use strict'

const state = Object.freeze({
  LOADING: 'loading',
  LISTENING: 'listening',
  PROCESSING: 'processing',
  IDLE: 'idle',
  PAUSED: 'paused',
  STOPPED: 'stopped'
})

class MockedBinding {
  constructor () {
    this._handle = null
    this._state = state.LOADING
    this._busy = false
    this._runToken = 0
    this._interfaceType = null
  }

  createInstance (interfaceType, configurationParams, outputCb, transitionCb = null) {
    console.log('Constructing the parakeet addon (ggml backend)')
    this._interfaceType = interfaceType
    this._config = configurationParams
    this.outputCb = outputCb
    this.transitionCb = transitionCb
    this._handle = { id: Date.now() }
    return this._handle
  }

  _callCallbacks (event, output, error = null) {
    if (this.outputCb) {
      this.outputCb(this, event, output, error)
    }
  }

  loadWeights (handle, data) {
    if (handle !== this._handle) throw new Error('Invalid handle')
    // Real binding accepts a single GGUF byte stream (`{ filename,
    // chunk, completed }`). The mock just logs the filename to
    // confirm the right shape was passed.
    console.log(`Loading GGUF: ${data?.filename || '<inline>'}`)
    return true
  }

  activate (handle) {
    if (handle !== this._handle) throw new Error('Invalid handle')
    console.log('Activated the addon')
    this._state = state.LISTENING
    this._busy = false
    if (this.transitionCb) {
      this.transitionCb(this, this._state)
    }
  }

  pause (handle) {
    if (handle !== this._handle) throw new Error('Invalid handle')
    console.log('Paused the processing')
    this._state = state.PAUSED
    if (this.transitionCb) {
      this.transitionCb(this, this._state)
    }
  }

  stop (handle) {
    if (handle !== this._handle) throw new Error('Invalid handle')
    console.log('Stopped the processing')
    this._state = state.STOPPED
    if (this.transitionCb) {
      this.transitionCb(this, this._state)
    }
  }

  cancel (handle) {
    if (handle !== this._handle) throw new Error('Invalid handle')
    console.log('Cancel job')
    this._runToken++
    this._busy = false
    this._state = state.LISTENING
    if (this.transitionCb) {
      this.transitionCb(this, this._state)
    }
  }

  status (handle) {
    if (handle !== this._handle) throw new Error('Invalid handle')
    return this._state
  }

  runJob (handle, data) {
    if (handle !== this._handle) throw new Error('Invalid handle')
    if (this._busy) {
      return false
    }

    if (data.type !== 'audio' || !data.input) {
      process.nextTick(() => {
        this._callCallbacks('Error', undefined, `Invalid runJob payload type: ${data.type}`)
      })
      return true
    }

    this._busy = true
    this._state = state.PROCESSING
    if (this.transitionCb) this.transitionCb(this, this._state)

    const runToken = ++this._runToken
    process.nextTick(() => {
      if (runToken !== this._runToken) return

      const audioLength = data.input.length ?? (data.input.byteLength / 4)
      const mockTranscription = {
        text: audioLength > 0 ? `Mock transcription for ${audioLength} samples of audio` : '[No speech detected]',
        start: 0,
        end: audioLength / 16000,
        toAppend: true
      }

      this._callCallbacks('Output', [mockTranscription], null)
      // Mirror the realistic key set ParakeetModel::runtimeStats()
      // emits (see addon/src/model-interface/parakeet/ParakeetModel.cpp)
      // so tests inspecting stats see the GGUF-backend shape.
      const audioDurationMs = Math.floor((audioLength / 16000) * 1000)
      this._callCallbacks('RuntimeStats', {
        processCalls: 1,
        totalSamples: audioLength,
        totalTokens: 0,
        totalTranscriptions: 1,
        totalWallMs: 1,
        totalTime: 1,
        modelLoadMs: 0,
        encoderMs: 1,
        decoderMs: 0,
        melSpecMs: 0,
        totalEncodedFrames: 0,
        audioDurationMs,
        backendDevice: 0,
        backendId: 0
      }, null)
      this._busy = false
      this._state = state.LISTENING
      if (this.transitionCb) this.transitionCb(this, this._state)
    })

    return true
  }

  load (handle, configurationParams) {
    if (handle !== this._handle) throw new Error('Invalid handle')
    console.log('Loaded configuration:', configurationParams)
    this._state = state.LOADING
    if (this.transitionCb) {
      this.transitionCb(this, this._state)
    }
  }

  reload (handle, configurationParams) {
    if (handle !== this._handle) throw new Error('Invalid handle')
    console.log('Reloaded configuration:', configurationParams)
    this._runToken++
    this._busy = false
    this._state = state.LOADING
    if (this.transitionCb) {
      this.transitionCb(this, this._state)
    }
    // After reload completes, transition back to IDLE to match C++ behavior
    process.nextTick(() => {
      this._state = state.IDLE
      if (this.transitionCb) {
        this.transitionCb(this, this._state)
      }
    })
  }

  unload (handle) {
    if (handle !== this._handle) throw new Error('Invalid handle')
    console.log('Unloaded the addon')
    this._state = state.IDLE
    if (this.transitionCb) {
      this.transitionCb(this, this._state)
    }
  }

  setLogger (callback) {
    console.log('Set logger')
  }

  releaseLogger () {
    console.log('Released logger')
  }

  unloadWeights (handle) {
    if (handle !== this._handle) throw new Error('Invalid handle')
    console.log('Unloaded weights')
    return true
  }

  destroyInstance (handle) {
    if (handle !== this._handle) throw new Error('Invalid handle')
    this._runToken++
    this._busy = false
    this._handle = null
    console.log('Destroyed the addon')
    this._state = state.IDLE
    if (this.transitionCb) {
      this.transitionCb(this, this._state)
    }
  }
}

module.exports = MockedBinding
