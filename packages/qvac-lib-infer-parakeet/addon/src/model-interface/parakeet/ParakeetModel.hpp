#pragma once

// Pure-ggml backend for the Parakeet binding (sourced from qvac-parakeet.cpp).
//
// This class used to host four onnxruntime sessions (preprocessor + encoder
// + decoder + ctc/sortformer) plus a hand-rolled mel-spectrogram, CMVN,
// chunked-limited streaming state machine for EOU, and a Sortformer
// post-processing pipeline. All of that has been replaced by a single
// `qvac_parakeet::Engine` from `parakeet-cpp` (vcpkg overlay port). The
// engine internally handles mel + encoder + decoder + diarization for any
// of the four model types (CTC, TDT, EOU, Sortformer) given a single GGUF
// file, so the binding's job is reduced to:
//
//   1. accumulate GGUF bytes from `setWeightsForFile()` into a temp file,
//   2. open `qvac_parakeet::Engine` against that path,
//   3. dispatch `process()` to either `transcribe_samples()` (CTC / TDT /
//      EOU) or `diarize_samples()` (Sortformer),
//   4. wrap the engine result in `Transcript` and fire the on-segment
//      callback.
//
// The public surface (constructor, `load()`, `process(any)`, `cancel()`,
// `setOnSegmentCallback()`, ...) is unchanged so existing JS callers keep
// working modulo the model-files change (`.gguf` instead of an ONNX dir).

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <streambuf>
#include <string>
#include <type_traits>
#include <vector>

#include "ParakeetConfig.hpp"
#include "model-interface/ParakeetTypes.hpp"
#include "qvac-lib-inference-addon-cpp/ModelInterfaces.hpp"
#include "qvac-lib-inference-addon-cpp/RuntimeStats.hpp"

namespace qvac_parakeet {
class Engine;
class StreamSession;
class SortformerStreamSession;
}

namespace qvac_lib_infer_parakeet {

class ParakeetModel : public qvac_lib_inference_addon_cpp::model::IModel,
                      public qvac_lib_inference_addon_cpp::model::IModelCancel,
                      public qvac_lib_inference_addon_cpp::model::IModelAsyncLoad {
public:
  using OutputCallback = std::function<void(const Transcript&)>;
  using ValueType = float;
  using Input = std::vector<ValueType>;
  using InputView = std::span<const ValueType>;
  using Output = std::vector<Transcript>;
  struct AnyInput {
    Input input;
  };

  explicit ParakeetModel(const ParakeetConfig& config);
  ~ParakeetModel();

  ParakeetModel(const ParakeetModel&) = delete;
  ParakeetModel& operator=(const ParakeetModel&) = delete;
  ParakeetModel(ParakeetModel&&) = delete;
  ParakeetModel& operator=(ParakeetModel&&) = delete;

  // ── Lifecycle ──────────────────────────────────────────────────────────
  void initializeBackend();
  void load();
  void unload();
  void unloadWeights() { unload(); }
  void reload();
  void reset();
  // Finalises the streaming session (if open) so the trailing partial
  // chunk's segments are flushed via the on-segment callback before the
  // session is torn down on unload(). For offline mode this is just a
  // flag flip.
  void endOfStream();
  bool isStreamEnded() const { return stream_ended_; }
  bool isLoaded() const { return is_loaded_; }
  qvac_lib_inference_addon_cpp::RuntimeStats runtimeStats() const override;
  std::any process(const std::any& input) override;
  std::string getName() const override;
  void cancel() const override;
  void warmup();

  // ── Processing ─────────────────────────────────────────────────────────
  void process(const Input& input);
  Output
  process(const Input& input, std::function<void(const Output&)> callback);

  // ── Configuration ──────────────────────────────────────────────────────
  void setConfig(const ParakeetConfig& config) { cfg_ = config; }
  void setOnSegmentCallback(const OutputCallback& callback) {
    on_segment_ = callback;
  }
  void addTranscription(const Transcript& transcript) {
    output_.push_back(transcript);
  }

  void saveLoadParams(const ParakeetConfig& config) { cfg_ = config; }

  template <typename T, typename... Args>
  typename std::enable_if<
      !std::is_same<typename std::decay<T>::type, ParakeetConfig>::value,
      void>::type
  saveLoadParams(T&&, Args&&...) {}

  void setWeightsForFile(
      const std::string& filename,
      std::unique_ptr<std::basic_streambuf<char>>&& streambuf) override;
  void waitForLoadInitialization() override { load(); }

  // Two streaming overloads -- mirror the legacy onnx binding's API so
  // existing JS callers' chunk-style file delivery keeps working without
  // changes. The ggml backend doesn't actually care about chunking; it
  // just buffers the bytes until `completed=true`, then materialises them
  // into a temp file on `load()`.
  void set_weights_for_file(
      const std::string& filename, std::span<const uint8_t> contents,
      bool completed);

  void set_weights_for_file(
      const std::string& filename,
      std::unique_ptr<std::basic_streambuf<char>> streambuf);

  template <typename T>
  void set_weights_for_file(const std::string& filename, T&& contents) {}

  // ── Queries ────────────────────────────────────────────────────────────
  [[nodiscard]] std::string getDisplayName() const { return getName(); }

  // Convenience helper -- decode raw int16 PCM bytes into normalised
  // float samples. Kept for back-compat with callers that used to pipe
  // raw mic captures straight into `process()`.
  [[nodiscard]] static std::vector<float> preprocessAudioData(
      const std::vector<uint8_t>& audioData,
      const std::string& audioFormat = "s16le");

private:
  void throwIfCancelled() const;
  static bool isCancellationError(const std::exception& e);

  // ── GGUF buffer staging ────────────────────────────────────────────────
  // The addon framework streams the GGUF bytes via setWeightsForFile().
  // We accumulate them into `gguf_buffer_` keyed by the (single) GGUF
  // filename; on load() we materialise the buffer into a temp file and
  // hand the path to qvac_parakeet::Engine.
  std::string                          gguf_filename_;
  std::vector<uint8_t>                 gguf_buffer_;
  std::filesystem::path                gguf_temp_path_;
  bool                                 gguf_completed_ = false;

  std::filesystem::path                writeBufferToTempFile_();
  void                                 cleanupTempFile_();

  // ── State ──────────────────────────────────────────────────────────────
  ParakeetConfig                       cfg_;
  OutputCallback                       on_segment_;
  Output                               output_;

  bool                                 stream_ended_ = false;
  bool                                 is_loaded_    = false;
  bool                                 is_warmed_up_ = false;

  // The Engine itself (pimpl-owned via unique_ptr to keep the
  // qvac-parakeet headers out of the binding's public include surface).
  std::unique_ptr<qvac_parakeet::Engine> engine_;
  mutable std::mutex                     engine_mutex_;

  // Streaming sessions (only one of the two is open at a time, depending on
  // model_type). Lifetime: opened in load() when cfg_.streaming == true,
  // finalize()d on endOfStream(), reset on unload(). Each process() call
  // routes through feed_pcm_f32() instead of the offline *_samples paths.
  std::unique_ptr<qvac_parakeet::StreamSession>           asr_session_;
  std::unique_ptr<qvac_parakeet::SortformerStreamSession> diar_session_;

  // Wall-clock seconds of audio fed to the streaming sessions so far,
  // used to translate per-session relative segment timestamps into a
  // monotonically growing wall-clock-style timeline that mirrors what
  // the offline path emits in `process(input)`.
  double                              streaming_audio_seconds_ = 0.0;
  bool                                streaming_finalized_     = false;

  // Sample rate in Hz; copied from cfg_.sampleRate at load time. The
  // ggml engine does not currently support non-16 kHz models, so anything
  // other than 16 000 throws on load.
  int                                  sample_rate_ = 16000;

  // ── Token / sentinel constants ─────────────────────────────────────────
  // These match the legacy onnx binding so JS-side string parsers don't
  // need to change. The engine itself uses different vocab IDs internally;
  // we surface only the "[No speech detected]" / "[Audio too short]" /
  // ... text sentinels through Transcript::text.
  static constexpr const char* ERR_NO_SPEECH        = "[No speech detected]";
  static constexpr const char* ERR_AUDIO_SHORT      = "[Audio too short]";
  static constexpr const char* ERR_MODEL_NOT_READY  = "[Model not ready]";
  static constexpr const char* ERR_MODEL_NOT_LOADED = "[Model not loaded]";
  static constexpr const char* ERR_INFERENCE       = "[Inference error]";
  static constexpr const char* ERR_NO_SPEAKERS     = "[No speakers detected]";
  static constexpr const char* ERR_JOB_CANCELLED   = "Job cancelled";

  static bool isSentinel(const std::string& text) {
    return text == ERR_NO_SPEECH || text == ERR_AUDIO_SHORT ||
           text == ERR_MODEL_NOT_READY || text == ERR_MODEL_NOT_LOADED ||
           text == ERR_INFERENCE || text == ERR_NO_SPEAKERS;
  }

  // ── Audio constants ────────────────────────────────────────────────────
  // The Engine handles its own mel-spectrogram internally; these are
  // here only so JS-facing logging / metric reporting keeps the same
  // numbers as the old binding.
  static constexpr int   HOP_LENGTH  = 160;
  static constexpr float SAMPLE_RATE = 16000.0f;

  DiarizationConfig                    diarConfig_;

  // ── Sortformer head dispatch ───────────────────────────────────────────
  std::string runSortformerProcess_(const Input& input);

  // ── ASR head dispatch ──────────────────────────────────────────────────
  std::string runAsrProcess_(const Input& input);

  // ── Streaming session helpers ──────────────────────────────────────────
  // Opens an ASR or Sortformer streaming session against the loaded engine.
  // Called from load() when cfg_.streaming == true. The on_segment callback
  // pushes a Transcript onto pending_streaming_segments_ for the next
  // process() call to drain into output_ + on_segment_.
  void openStreamingSession_();
  void closeStreamingSession_();

  // process() drainage: streaming-session callbacks fire mid-feed (and
  // potentially from a different thread on finalize()), so we stash the
  // per-segment Transcripts here under streaming_mutex_ and flush them
  // into output_ at the end of each process() call.
  std::mutex                          streaming_mutex_;
  std::vector<Transcript>             pending_streaming_segments_;

  // Runs cfg_.streaming feed for a chunk and returns the concatenated
  // text of the segments fired during the call (joined with single
  // spaces). Sentinel-string fallbacks ([No speech detected] etc.) are
  // applied when the session emitted nothing for the chunk so the
  // existing Transcript-shaped JS contract stays intact.
  std::string runStreamingProcess_(const Input& input);

  // ── Runtime stats (subset of legacy fields; we now derive most numbers
  //     from the Engine's own per-call timings) ────────────────────────
  float                                processed_time_       = 0.0f;
  int64_t                              totalSamples_         = 0;
  int64_t                              totalTokens_          = 0;
  int64_t                              totalTranscriptions_  = 0;
  int64_t                              processCalls_         = 0;
  int64_t                              totalWallMs_          = 0;
  int64_t                              modelLoadMs_          = 0;
  int64_t                              melSpecMs_            = 0;
  int64_t                              encoderMs_            = 0;
  int64_t                              decoderMs_            = 0;
  int64_t                              totalMelFrames_       = 0;
  int64_t                              totalEncodedFrames_   = 0;

  mutable std::atomic_uint64_t         nextGeneration_   = 1;
  mutable std::atomic_uint64_t         activeGeneration_ = 0;
  mutable std::atomic_uint64_t         cancelGeneration_ = 0;
};

} // namespace qvac_lib_infer_parakeet
