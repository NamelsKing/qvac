import QvacResponse from '@qvac/infer-base/src/QvacResponse'
import type { LoggerInterface } from '@qvac/logging'
import { Readable } from 'stream'

/**
 * Model type discriminator. The binding auto-detects this from the
 * loaded GGUF's `parakeet.model.type` metadata field; this type is
 * only here for callers that want to surface it in their own UI.
 */
declare type ModelType = 'tdt' | 'ctc' | 'eou' | 'sortformer'

/**
 * Parakeet-specific configuration options. The model type itself is
 * not configured here -- it's auto-detected from the GGUF metadata.
 */
declare interface ParakeetConfig {
  /** Maximum CPU threads for inference (0 lets the engine pick) */
  maxThreads?: number
  /** Enable the linked ggml GPU backend (Metal / Vulkan / CUDA) */
  useGPU?: boolean
  /** Audio sample rate in Hz (default: 16000; engine assumes 16 kHz) */
  sampleRate?: number
  /** Number of audio channels (default: 1, must be mono) */
  channels?: number
  /** Enable caption/subtitle mode (default: false) */
  captionEnabled?: boolean
  /** Include timestamps in output (default: true) */
  timestampsEnabled?: boolean
  /** Random seed for reproducibility (-1 for random, default: -1) */
  seed?: number

  /**
   * Open a long-lived streaming session (StreamSession for ASR,
   * SortformerStreamSession for diarization) at load() time and
   * route each `process()` call through `feed_pcm_f32()`. Speaker
   * IDs stay stable across appends, EOU `<EOU>` boundaries surface
   * as segment markers, and CTC/TDT can opt into energy-VAD events.
   * Default: false (offline `transcribe_samples` / `diarize_samples`).
   */
  streaming?: boolean
  /** Streaming chunk cadence in milliseconds (default: 2000) */
  streamingChunkMs?: number
  /** Sortformer rolling-history window in ms (default: 30000) */
  streamingHistoryMs?: number
  /** Emit partial segments before chunk boundaries (default: true) */
  streamingEmitPartials?: boolean
  /** CTC/TDT-only energy-VAD events (default: false) */
  streamingEnergyVad?: boolean
}

/**
 * Map of model file paths supplied to TranscriptionParakeet.
 */
declare interface TranscriptionParakeetFiles {
  /**
   * Absolute path to a single `.gguf` checkpoint produced by
   * `qvac-parakeet.cpp/scripts/convert-nemo-to-gguf.py`. The same
   * field accepts CTC, TDT, EOU, and Sortformer GGUFs -- the binding
   * picks the right dispatch from the file's metadata.
   */
  model?: string
}

/**
 * Options accepted by the TranscriptionParakeet constructor.
 */
declare interface TranscriptionParakeetArgs {
  files?: TranscriptionParakeetFiles
  config?: TranscriptionParakeetConfig
  logger?: LoggerInterface
  exclusiveRun?: boolean
  [key: string]: unknown
}

/**
 * Configuration for TranscriptionParakeet (non-path settings only).
 */
declare interface TranscriptionParakeetConfig {
  enableStats?: boolean
  parakeetConfig?: ParakeetConfig
  [key: string]: unknown
}

/**
 * Transcription segment returned by the model.
 */
declare interface TranscriptionSegment {
  text: string
  start: number
  end: number
  toAppend: boolean
  id?: number
}

/**
 * Output callback events.
 */
declare type OutputEvent = 'JobStarted' | 'Output' | 'JobEnded' | 'Error'

/**
 * Input types accepted by the Parakeet addon.
 */
declare type AppendInput =
  | { type: 'audio'; data: ArrayBuffer; priority?: number }
  | { type: 'end of job' }

/**
 * Minimal interface for the native addon.
 */
declare interface Addon {
  activate(): Promise<void>
  /** Returns the JS-owned job ID for the buffered or running transcription. */
  append(input: AppendInput): Promise<number>
  /** Cancels the matching JS-owned job when one is active or buffered. */
  cancel(jobId?: number): Promise<void>
  loadWeights(weightsData: { filename: string; chunk: Uint8Array; completed: boolean }): Promise<void>
  status(): Promise<string>
  pause(): Promise<void>
  stop(): Promise<void>
  reload(config: ParakeetConfig): Promise<void>
  destroyInstance(): Promise<void>
}

declare interface InferenceClientState {
  configLoaded: boolean
  weightsLoaded: boolean
  destroyed: boolean
}

/**
 * High-level Parakeet speech-to-text client backed by the ggml engine
 * sourced from qvac-parakeet.cpp. Accepts a single `.gguf` checkpoint
 * (CTC / TDT / EOU / Sortformer) -- the binding auto-detects the
 * model type from GGUF metadata.
 */
declare class TranscriptionParakeet {
  protected readonly _config: TranscriptionParakeetConfig
  protected addon: Addon
  protected params: ParakeetConfig

  constructor(opts: TranscriptionParakeetArgs)

  validateModelFiles(): void
  protected _load(): Promise<void>
  load(): Promise<void>

  /**
   * Run inference on an audio stream. When `opts.stats` was set on
   * construction, `response.stats` matches {@link TranscriptionParakeet.RuntimeStats}.
   */
  run(
    audioStream: Readable
  ): Promise<QvacResponse<TranscriptionParakeet.ParakeetRunOutput>>

  reload(newConfig?: { parakeetConfig?: Partial<ParakeetConfig> }): Promise<void>
  unload(): Promise<void>
  getState(): InferenceClientState
  cancel(): Promise<void>
  status(): Promise<string | undefined>
  pause(): Promise<void>
  unpause(): Promise<void>
  destroy(): Promise<void>
}

declare namespace TranscriptionParakeet {
  /**
   * Keys returned by the native addon `ParakeetModel::runtimeStats()`
   * when stats are enabled. `totalTime` and `totalWallMs` are wall
   * time in milliseconds; `audioDurationMs` and other `*Ms` fields
   * are milliseconds where applicable. `decoderMs`, `melSpecMs`,
   * `totalEncodedFrames`, and `totalTokens` are populated only by
   * the offline ASR path and stay 0 for streaming / Sortformer.
   */
  export interface RuntimeStats {
    totalTime: number
    audioDurationMs: number
    totalSamples: number
    totalTokens: number
    totalTranscriptions: number
    processCalls: number
    modelLoadMs: number
    melSpecMs: number
    encoderMs: number
    decoderMs: number
    totalWallMs: number
    totalEncodedFrames: number
  }

  /**
   * Payload passed to `onUpdate` (array of segments or a single segment).
   */
  export type ParakeetRunOutput = TranscriptionSegment[] | TranscriptionSegment

  export {
    TranscriptionParakeet as default,
    TranscriptionParakeet,
    ModelType,
    ParakeetConfig,
    TranscriptionParakeetFiles,
    TranscriptionParakeetArgs,
    TranscriptionParakeetConfig,
    TranscriptionSegment,
    OutputEvent,
    AppendInput,
    Addon,
    InferenceClientState
  }
}

export = TranscriptionParakeet
