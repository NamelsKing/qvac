#pragma once

#include <string>

#include "model-interface/ParakeetTypes.hpp"

namespace qvac_lib_infer_parakeet {

struct ParakeetConfig {
  std::string modelPath;

  // ModelType is auto-detected by ParakeetModel::load() from the
  // engine's `parakeet.model.type` GGUF metadata; the default below
  // is just a placeholder until that override fires.
  ModelType modelType = ModelType::TDT;

  int  maxThreads        = 4;
  bool useGPU            = false;
  int  sampleRate        = 16000;
  int  channels          = 1;
  bool captionEnabled    = false;
  bool timestampsEnabled = true;
  int  seed              = -1;

  // ── Streaming mode ──────────────────────────────────────────────────────
  // When true, the model opens a long-lived qvac_parakeet streaming session
  // (StreamSession for ASR, SortformerStreamSession for diarization) at
  // load() time and routes each process() call through feed_pcm_f32(). The
  // session retains state (KV cache for ASR Mode 3, rolling speaker history
  // for Sortformer) across appends, so:
  //   - Sortformer speaker IDs stay stable from chunk to chunk.
  //   - EOU `<EOU>` boundaries surface as segment markers (and StreamEvents).
  //   - Optional energy-VAD events fire for CTC/TDT.
  // Off by default for batch-style transcription.
  bool streaming             = false;
  int  streamingChunkMs      = 2000;
  int  streamingHistoryMs    = 30000;   // Sortformer rolling window only
  bool streamingEmitPartials = true;
  bool streamingEnergyVad    = false;   // CTC/TDT only; ignored elsewhere

  ParakeetConfig() = default;
  explicit ParakeetConfig(const std::string& path) : modelPath(path) {}

  bool operator==(const ParakeetConfig& other) const {
    return modelPath == other.modelPath &&
           modelType == other.modelType && maxThreads == other.maxThreads &&
           useGPU == other.useGPU && sampleRate == other.sampleRate &&
           channels == other.channels &&
           captionEnabled == other.captionEnabled &&
           timestampsEnabled == other.timestampsEnabled && seed == other.seed &&
           streaming == other.streaming &&
           streamingChunkMs == other.streamingChunkMs &&
           streamingHistoryMs == other.streamingHistoryMs &&
           streamingEmitPartials == other.streamingEmitPartials &&
           streamingEnergyVad == other.streamingEnergyVad;
  }

  bool operator!=(const ParakeetConfig& other) const { return !(*this == other); }
};

} // namespace qvac_lib_infer_parakeet
