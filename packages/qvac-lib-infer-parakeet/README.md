# qvac-lib-infer-parakeet

This library simplifies running NVIDIA Parakeet speech-to-text and Sortformer speaker-diarization inference within QVAC runtime applications. It provides an easy interface to load, execute, and manage Parakeet inference instances, supporting CTC, TDT, EOU, and Sortformer checkpoints from a single binding.

**Note: This library uses the [`qvac-parakeet.cpp`](https://github.com/GustavoA1604/qvac-parakeet.cpp) ggml engine for inference. The previous onnxruntime-based implementation has been replaced.**

## Table of Contents

- [Supported Platforms](#supported-platforms)
- [Installation](#installation)
- [Development](#development)
- [Usage](#usage)
  - [1. Stage a Model](#1-stage-a-model)
  - [2. Configure the Model](#2-configure-the-model)
  - [3. Create Model Instance](#3-create-model-instance)
  - [4. Load the Model](#4-load-the-model)
  - [5. Run Inference](#5-run-inference)
  - [6. Release Resources](#6-release-resources)
- [Quickstart example](#quickstart-example)
- [Model Variants](#model-variants)
- [Other examples](#other-examples)
- [Glossary](#glossary)
- [Error Range](#error-range)
- [Resources](#resources)
- [License](#license)

## Supported Platforms

| Platform | Architecture | Min Version | Status | GPU Support |
|----------|-------------|-------------|--------|-------------|
| macOS | arm64, x64 | 14.0+ | Tier 1 | Metal |
| iOS | arm64 | 17.0+ | Tier 1 | Metal |
| Linux | arm64, x64 | Ubuntu-22+ | Tier 1 | Vulkan |
| Android | arm64 | 12+ | Tier 1 | Vulkan / OpenCL |
| Windows | x64 | 10+ | Tier 1 | Vulkan |

**Dependencies:**
- qvac-lib-inference-addon-cpp: C++ addon framework
- parakeet-cpp (latest): inference engine, sourced from
  [`qvac-parakeet.cpp`](https://github.com/GustavoA1604/qvac-parakeet.cpp);
  bundles ggml at the pinned upstream commit
- Bare Runtime (latest): JavaScript runtime
- Linux requires Clang/LLVM 19 with libc++

## Installation

### Prerequisites

Make sure [Bare](#glossary) Runtime is installed:
```bash
npm install -g bare bare-make
```

### Installing the Package

Install the latest version:
```bash
npm install @qvac/transcription-parakeet@latest
```

## Development

### Building the AddOn Locally

For local development, you'll need to build the native addon that interfaces with the Parakeet engine. Follow these steps:

#### Prerequisites

First, make sure you have the prerequisites from the [Installation](#installation) section.

#### System Requirements

**Supported Platforms:**
- **Linux** (x64, ARM64) -- requires Clang/LLVM 19 with libc++
- **macOS** (x64, ARM64)
- **Windows** (x64)

#### Required Tools

**All Platforms:**
- **CMake** (>= 3.25)
- **Git**
- **C++ Compiler** with C++20 support
  - Linux: Clang 19+ with libc++
  - macOS: Xcode 12+ (provides Clang 12+)
  - Windows: Visual Studio 2019+ or MinGW-w64

#### vcpkg Setup

This project uses [vcpkg](https://vcpkg.io/) for C++ dependency management. The `cmake-vcpkg` package pulls vcpkg in transparently during `npm install`, so most users don't need to set it up by hand. If you want a system-wide vcpkg checkout:

```bash
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh           # or .\bootstrap-vcpkg.bat on Windows
export VCPKG_ROOT=$(pwd)
```

#### Platform-Specific Setup

**Linux:**
```bash
# Ubuntu/Debian -- includes Clang 19 and libc++ required by the native addon
sudo apt update
sudo apt install clang libc++-dev libc++abi-dev build-essential cmake git pkg-config
```

**macOS:**
```bash
xcode-select --install
brew install cmake git
```

**Windows:**
- Install [Visual Studio 2019+](https://visualstudio.microsoft.com/) with C++ development tools
- Install [CMake](https://cmake.org/download/) (3.25+)
- Install [Git for Windows](https://git-scm.com/download/win)

#### GPU Acceleration (Optional)

GPU backends are selected at vcpkg install time via the `parakeet-cpp[metal|vulkan|cuda|opencl]` features. The bundled ggml inside the `parakeet-cpp` port handles backend wiring; runtime falls back to CPU if the chosen backend doesn't initialise.

- **Metal (macOS/iOS):** automatic; no setup required.
- **Vulkan (Linux/Windows/Android):** install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) and ensure GPU drivers support Vulkan 1.1+.
  ```bash
  # Ubuntu/Debian
  sudo apt install vulkan-tools libvulkan-dev vulkan-utility-libraries-dev spirv-tools
  ```

#### Clone and Setup

```bash
git clone https://github.com/tetherto/qvac.git
cd qvac/packages/qvac-lib-infer-parakeet
npm install
```

#### Build the Native AddOn

```bash
npm run build
```

This runs:
1. `bare-make generate` -- generates build configuration
2. `bare-make build` -- compiles the native C++ addon
3. `bare-make install` -- installs the prebuild

#### Running Tests

```bash
npm run test:unit                                    # JS unit tests (mocked)
QVAC_TEST_GGUF_DIR=models npm run test:integration   # JS integration vs. real GGUFs
npm run test:cpp                                     # gtest C++ suite
```

The integration suite locates each model type via `QVAC_TEST_GGUF_DIR=<path-with-staged-ggufs>` (or per-model overrides like `QVAC_TEST_GGUF_TDT=/full/path.gguf`). Tests skip cleanly when no GGUF is available, so CI without local models still passes.

## Usage

The library wraps `qvac-parakeet.cpp`'s engine in the QVAC addon framework so you can transcribe audio files, run speaker diarization, or stream live mic input through the same shape: load a single `.gguf`, push audio chunks, drain segment callbacks.

> **Heads up:** the package is intended to be used through `index.js`'s `TranscriptionParakeet` class for typical apps, or through the lower-level `parakeet.js` `ParakeetInterface` when you need direct control of the lifecycle (as the bundled examples do).

### 1. Stage a Model

The ggml backend takes a single `.gguf` per checkpoint. The standard flow is "download `.nemo` from HuggingFace, convert to `.gguf` via `qvac-parakeet.cpp`'s converter":

```bash
npm run setup-models                       # downloads + converts all 4 models, q8_0
npm run setup-models -- -t tdt             # just TDT
npm run setup-models -- -t eou -q f16      # full-precision EOU
```

Output GGUFs land in `./models/`. The conversion step uses `qvac-parakeet.cpp`'s Python venv automatically when a sibling `~/dev/qvac-parakeet.cpp` checkout is present; otherwise pass `--python /path/to/venv/bin/python` (NeMo + `gguf` + numpy + torch required).

The two underlying scripts are also flag-driven if you want to run them separately:

```
download-models.sh [--type ctc|tdt|eou|sortformer|all]
                   [--output <dir>] [--force] [--help]
convert-nemo.sh    [--type ctc|tdt|eou|sortformer|all]
                   [--quant f16|q8_0|q5_0|q4_0|f32]
                   [--parakeet-cpp <path>] [--python <bin>]
                   [--nemo-dir <dir>] [--output <dir>] [--force] [--help]
```

#### Source repositories

| Model | HuggingFace `.nemo` |
|-------|-----------------------------------|
| CTC | [`nvidia/parakeet-ctc-0.6b`](https://huggingface.co/nvidia/parakeet-ctc-0.6b) |
| TDT | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| EOU | [`nvidia/parakeet_realtime_eou_120m-v1`](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1) |
| Sortformer | [`nvidia/diar_sortformer_4spk-v1`](https://huggingface.co/nvidia/diar_sortformer_4spk-v1) |

NVIDIA Open Model License -- see each repo's model card for terms.

### 2. Configure the Model

Most users interact with the package through `index.js`. From that entrypoint we surface a small, safe subset of options; the rest keep `parakeet-cpp` defaults.

#### What `index.js` accepts

| Section | Key | Description |
| --- | --- | --- |
| `files` | `model` | Absolute or relative path to the `.gguf` checkpoint |
| `config.parakeetConfig` | `maxThreads` | CPU threads; `0` lets the engine pick `hardware_concurrency` |
| | `useGPU` | Enable the linked ggml GPU backend (default: `false`) |
| | `streaming` | Open a long-lived `StreamSession` / `SortformerStreamSession` so speaker IDs stay stable across appends and EOU `<EOU>` boundaries surface as segments. Default: `false` (offline `transcribe_samples` / `diarize_samples`). |
| | `streamingChunkMs` | Streaming chunk cadence in ms (default: 2000) |
| | `streamingHistoryMs` | Sortformer rolling-history window in ms (default: 30000) |
| | `streamingEmitPartials` | Emit partials before chunk boundaries (default: `true`) |
| | `streamingEnergyVad` | CTC/TDT energy-VAD events (default: `false`) |

The model type (CTC / TDT / EOU / Sortformer) is **auto-detected from the GGUF metadata**, so callers don't need to pass `modelType`. Other knobs (`captionEnabled`, `timestampsEnabled`, `seed`, `sampleRate`, `channels`) keep sensible defaults.

#### Configuration Example

```javascript
const config = {
  parakeetConfig: {
    useGPU:    true,
    streaming: false   // flip to true for live-mic / speaker-stable streaming
  }
}
```

### 3. Create Model Instance

```javascript
const TranscriptionParakeet = require('@qvac/transcription-parakeet')

const model = new TranscriptionParakeet({
  files: { model: './models/parakeet-tdt-0.6b-v3.q8_0.gguf' },
  config: {
    parakeetConfig: { useGPU: true }
  }
})
```

### 4. Load the Model

```javascript
try {
  await model.load()
} catch (error) {
  console.error('Failed to load model:', error)
}
```

`load()` opens the `.gguf`, instantiates `qvac_parakeet::Engine`, and (if `streaming: true`) opens the relevant streaming session.

### 5. Run Inference

Pass an audio stream (e.g. from `bare-fs.createReadStream` or a live PCM buffer) to `run()`. Audio must be **16 kHz mono**, either Float32 or signed 16-bit little-endian PCM.

There are two ways to receive transcription results:

#### Option 1: Real-time streaming with `onUpdate()`

```javascript
try {
  const audioStream = fs.createReadStream('path/to/audio.raw', {
    highWaterMark: 16000
  })
  const response = await model.run(audioStream)

  await response
    .onUpdate(segments => {
      // `segments` is `Array<{ text, start, end, toAppend, id }>`
      for (const seg of segments) console.log(seg.text)
    })
    .await()
} catch (error) {
  console.error('Transcription failed:', error)
}
```

#### Option 2: Complete result with `iterate()`

```javascript
const response = await model.run(audioStream)
for await (const chunk of response.iterate()) {
  console.log('Transcription chunk:', chunk)
}
```

**Key differences:**
- `onUpdate()` -- segments arrive as the engine produces them (chunk by chunk for the streaming session, or as the offline encoder fires its callback).
- `iterate()` -- collects all segments after the job finishes.

For Sortformer GGUFs, the same `Output` event carries `Speaker N: HH:MM:SS - HH:MM:SS` text per segment instead of an ASR transcript -- see `examples/diarized-transcribe.js` for parsing.

### 6. Release Resources

```javascript
try {
  await model.unload()
} catch (error) {
  console.error('Failed to unload model:', error)
}
```

## Quickstart example

### 1. Clone the repo & install dependencies

```bash
git clone https://github.com/tetherto/qvac.git
cd qvac/packages/qvac-lib-infer-parakeet
npm install
```

`npm install` pulls the `parakeet-cpp` overlay port (which bundles ggml at the pinned upstream commit) and produces `prebuilds/<platform>-<arch>/qvac__transcription-parakeet.bare`.

### 2. Stage a model

```bash
npm run setup-models -- -t tdt -q q8_0
```

### 3. Run the bundled examples

```bash
# Single-file transcription (any model type -- CTC / TDT / EOU / Sortformer)
bare examples/transcribe.js \
     --model models/parakeet-tdt-0.6b-v3.q8_0.gguf \
     --audio examples/samples/sample-16k.wav

# Combined ASR + diarization
bare examples/diarized-transcribe.js \
     --asr-model  models/parakeet-tdt-0.6b-v3.q8_0.gguf \
     --diar-model models/sortformer-4spk-v1.q8_0.gguf \
     --audio      examples/samples/two-speakers-16k.wav

# Live mic transcription
bare examples/live-mic.js --model models/parakeet-eou-120m-v1.q8_0.gguf --accumulate

# Live mic + speaker tagging
bare examples/live-mic-diarized.js \
     --asr-model  models/parakeet-tdt-0.6b-v3.q8_0.gguf \
     --diar-model models/sortformer-4spk-v1.q8_0.gguf --accumulate
```

> If you use `npm run example:* -- ...` instead of `bare`, remember the `--` separator -- without it npm interprets `--model` as one of its own config flags.

The live-mic examples capture the default input device via `sox -d` (install: `brew install sox` / `apt install sox` / `choco install sox`). With `--accumulate`, transcripts append onto one line per turn and flush on silence, speaker change, or Ctrl-C.

## Model Variants

| Variant | Languages | Decoder | Default GGUF size (q8_0) | Notes |
|---------|-----------|---------|-------------------------:|-------|
| **CTC** | English | argmax CTC | ~ 700 MiB | Fast, no PnC. |
| **TDT** | ~25 | RNN-T greedy + duration | ~ 715 MiB | Recommended default; PnC + auto-detect. |
| **EOU** | English | RNN-T greedy + `<EOU>` | ~ 132 MiB | Streaming-trained; native end-of-turn token. |
| **Sortformer** | n/a | Diarization head | ~ 141 MiB | 4-speaker. |

Per-tier sizes (`f16` / `q8_0` / `q4_0`) are listed in the [`parakeet-cpp` README](https://github.com/GustavoA1604/qvac-parakeet.cpp#supported-checkpoints).

## Other examples

- [`examples/transcribe.js`](examples/transcribe.js) -- universal single-file transcribe / diarize (any GGUF, all model types).
- [`examples/diarized-transcribe.js`](examples/diarized-transcribe.js) -- combined Sortformer + ASR pipeline ("who said what").
- [`examples/live-mic.js`](examples/live-mic.js) -- live microphone transcription via `sox` and the streaming session.
- [`examples/live-mic-diarized.js`](examples/live-mic-diarized.js) -- live mic with parallel Sortformer + ASR for speaker-tagged transcripts.
- [`examples/decode-audio.js`](examples/decode-audio.js) -- decode + transcribe in one step. Same flag surface as `transcribe.js` but pipes the input through `@qvac/decoder-audio` (FFmpeg) first, so any container / codec FFmpeg supports (mp3, m4a, ogg, flac, mp4, ...) works -- not just 16 kHz mono `.wav` / raw s16le PCM.
- [`examples/utils.js`](examples/utils.js) -- shared helpers used by the examples (`loadWeights` streaming, `Output`/`JobEnded` race resolution).

## Glossary

- **Bare** -- small, modular JavaScript runtime for desktop and mobile. [Learn more](https://docs.pears.com/bare-reference/overview).
- **GGUF** -- single-file model format used by ggml-based runtimes; carries weights + tokenizer + hyperparameters in one file.
- **QVAC** -- our open-source AI-SDK for building decentralized AI applications.

## Error Range

All errors from this library are in the range of 24,001 to 25,000.

## Resources

- [`qvac-parakeet.cpp`](https://github.com/GustavoA1604/qvac-parakeet.cpp) -- the C++/ggml inference engine this binding wraps.
- [NVIDIA Parakeet model cards](https://huggingface.co/collections/nvidia/parakeet-asr-models-66b50d5a37b9580ee4ba93c2) -- upstream `.nemo` checkpoints.

## License

This project is licensed under the Apache-2.0 License -- see [LICENSE](LICENSE) for details. Model files are distributed under the **NVIDIA Open Model License**; see the upstream HuggingFace cards for the per-checkpoint terms.

For questions or issues, please open an issue on the GitHub repository.
