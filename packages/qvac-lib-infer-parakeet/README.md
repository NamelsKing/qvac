# qvac-lib-infer-parakeet

**Technology Stack:** C++20, CMake, vcpkg, Bare Runtime, ggml (parakeet-cpp)
**Package Type:** Native Bare addon

A high-performance speech-to-text addon for the Bare runtime using NVIDIA's
Parakeet ASR family. The inference engine is a pure-ggml backend sourced
from [`qvac-parakeet.cpp`](https://github.com/GustavoA1604/qvac-parakeet.cpp)
via its `parakeet-cpp` vcpkg overlay port -- no onnxruntime, no Python at
runtime, single self-contained `.gguf` per model.

## Key Features

- **Single-file GGUF models** -- one `.gguf` per checkpoint, statically
  linked into the addon binary alongside ggml. No `model.onnx_data`
  side-loads, no `tokenizer.json`/`vocab.txt` shimming -- the GGUF
  carries the tokenizer and all hyperparameters.
- **Four engines, one binding**:
  - **CTC** (English, fast)
  - **TDT** (~25 languages with PnC, recommended default)
  - **EOU** (real-time streaming; emits `<EOU>` end-of-utterance token)
  - **Sortformer** (4-speaker diarization)
- **GPU acceleration via ggml backends** (Metal on macOS/iOS, Vulkan on
  Linux/Windows/Android, OpenCL opt-in). The `parakeet-cpp` port handles
  backend selection at install time; runtime falls back to CPU if the
  selected backend doesn't initialise.
- **Quantised by default** -- recommended GGUFs ship at q8_0 (~1.9× smaller
  than f16, no user-visible transcript regression). q4_0 also available.
- **Job cancellation, async output stream, Bare-runtime-native**
  -- inherited from `qvac-lib-inference-addon-cpp`.

## Built With

- [`qvac-lib-inference-addon-cpp`](https://github.com/tetherto/qvac-lib-inference-addon-cpp)
  -- the QVAC addon framework (job runner, output callbacks, lifecycle).
- [`parakeet-cpp`](https://github.com/GustavoA1604/qvac-parakeet.cpp) --
  pure C++/ggml inference engine (FastConformer encoder + CTC/TDT/EOU/
  Sortformer decoders + the Phase 13 cross-engine StreamEvent API).
- [`ggml`](https://github.com/GustavoA1604/qvac-ext-ggml) -- pinned to
  upstream `ggml-org/ggml@58c38058` via the bundled-ggml branch on the
  Gustavo fork (no chatterbox patches).

## Table of Contents

- [Installation](#installation)
- [Model Setup](#model-setup)
- [Examples](#examples)
- [JavaScript API](#javascript-api)
- [Model Variants](#model-variants)
- [Development](#development)
- [Supported Platforms](#supported-platforms)
- [License](#license)

## Installation

### Prerequisites

- **Bare Runtime** -- install from
  [holepunchto/bare](https://github.com/holepunchto/bare).
- **Node.js / npm** (>= 18) for dependency install.
- **vcpkg** -- pulled in transparently by `cmake-vcpkg` during `npm install`.
- **C++ compiler** with C++20 support
  - macOS: Xcode Command Line Tools
  - Linux: Clang/LLVM 19 with libc++
  - Windows: Visual Studio 2022 + C++ workload

### Linux build prerequisites

```bash
sudo apt install clang libc++-dev libc++abi-dev build-essential pkg-config
```

### Build from source

```bash
git clone https://github.com/tetherto/qvac.git
cd qvac/packages/qvac-lib-infer-parakeet
npm install
```

`npm install` runs `bare-make generate && bare-make build && bare-make
install` for you. The `parakeet-cpp` overlay port (which bundles ggml at
the pinned upstream commit) installs into the local vcpkg root and the
addon binary lands in `prebuilds/<platform>-<arch>/`.

To rebuild after pulling changes:

```bash
npm run build
```

## Model Setup

The ggml backend takes a single `.gguf` per checkpoint.
Standard flow: download the upstream NeMo `.nemo` archive from
HuggingFace, then convert it locally with the `qvac-parakeet.cpp`
converter.

### One-shot setup (recommended)

```bash
npm run setup-models             # downloads + converts all 4 models, q8_0
npm run setup-models -- -t tdt   # just TDT
```

`setup-models` runs `download-models.sh && convert-nemo.sh` end to end.
Output GGUFs land in `./models/`.

### Step by step

```bash
npm run download-models                      # all four .nemo archives
npm run download-models -- -t eou            # just EOU
npm run convert-models                       # all q8_0
npm run convert-models -- -t tdt -q q4_0     # TDT q4_0 only
```

Both scripts are flag-driven:

```
download-models.sh [--type ctc|tdt|eou|sortformer|all]
                   [--output <dir>] [--force] [--help]
convert-nemo.sh    [--type ctc|tdt|eou|sortformer|all]
                   [--quant f16|q8_0|q5_0|q4_0|f32]
                   [--parakeet-cpp <path>] [--python <bin>]
                   [--nemo-dir <dir>] [--output <dir>] [--force] [--help]
```

### Conversion prerequisites

`convert-nemo.sh` calls into qvac-parakeet.cpp's
`scripts/convert-nemo-to-gguf.py`, which needs Python with NeMo +
`gguf` + numpy + torch. The script auto-detects
`qvac-parakeet.cpp/venv/bin/python` if a sibling
`~/dev/qvac-parakeet.cpp` checkout exists; otherwise pass `--python
/path/to/your/venv/bin/python` or set `PYTHON=...`.

A pre-flight import check fails fast with a clear list of missing
modules instead of letting `python` crash mid-conversion.

### Source repositories

| Model | HuggingFace `.nemo` |
|-------|-----------------------------------|
| CTC | [`nvidia/parakeet-ctc-0.6b`](https://huggingface.co/nvidia/parakeet-ctc-0.6b) |
| TDT | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| EOU | [`nvidia/parakeet_realtime_eou_120m-v1`](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1) |
| Sortformer | [`nvidia/diar_sortformer_4spk-v1`](https://huggingface.co/nvidia/diar_sortformer_4spk-v1) |

NVIDIA Open Model License -- see each repo's model card for terms.

## Examples

After `npm install` + `npm run setup-models`, five example scripts
cover every supported workflow. All examples are minimal -- only the
required model and audio paths plus an `--accumulate` flag on the
live-mic ones. Type, threads, GPU, capture command, log level, etc.
are picked sensibly by default; tweak them by editing the example if
needed.

> If you prefer `npm run example:* -- ...` over calling `bare`
> directly, remember the `--` separator: `npm run example:mic --
> --model models/parakeet-eou-120m-v1.q8_0.gguf`. Without it, npm
> interprets `--model` as its own config flag.

### Transcribe / diarize a single file

```bash
bare examples/transcribe.js \
     --model models/parakeet-tdt-0.6b-v3.q8_0.gguf \
     --audio examples/samples/sample-16k.wav
```

Same script handles all four engines (CTC, TDT, EOU, Sortformer);
the binding reads `parakeet.model.type` from the GGUF metadata and
routes to the right pipeline.

### Diarized transcription ("who said what")

```bash
bare examples/diarized-transcribe.js \
     --asr-model  models/parakeet-tdt-0.6b-v3.q8_0.gguf \
     --diar-model models/sortformer-4spk-v1.q8_0.gguf \
     --audio      examples/samples/two-speakers-16k.wav
```

Runs Sortformer to find speaker time-segments, then transcribes each
slice with the ASR engine and emits a `Speaker N: ...` per-segment
transcript.

### Live microphone (transcription)

```bash
bare examples/live-mic.js --model models/parakeet-tdt-0.6b-v3.q8_0.gguf
bare examples/live-mic.js --model models/parakeet-eou-120m-v1.q8_0.gguf --accumulate
```

Captures the default input device via `sox -d` (install: `brew
install sox` / `apt install sox` / `choco install sox`) and streams
2-second chunks into the binding's streaming session. With
`--accumulate`, transcripts are appended onto one line per turn and
flushed on silence or shutdown -- matches the C++ live-mic example
in [qvac-parakeet.cpp](https://github.com/GustavoA1604/qvac-parakeet.cpp).
Press Ctrl-C to flush and exit.

### Live microphone + diarization

```bash
bare examples/live-mic-diarized.js \
     --asr-model  models/parakeet-tdt-0.6b-v3.q8_0.gguf \
     --diar-model models/sortformer-4spk-v1.q8_0.gguf
bare examples/live-mic-diarized.js \
     --asr-model  models/parakeet-tdt-0.6b-v3.q8_0.gguf \
     --diar-model models/sortformer-4spk-v1.q8_0.gguf \
     --accumulate
```

Same capture loop as `live-mic.js`, with a Sortformer
`SortformerStreamSession` running in parallel for stable speaker IDs
across chunks (the session's internal rolling-history window keeps
the 4-speaker permutation consistent across calls). Each printed
line is tagged with the dominant speaker for that chunk.

### Audio decoding

```bash
bare examples/decode-audio.js
```

Demonstrates `@qvac/decoder-audio` to decode arbitrary audio formats
(MP3, M4A, OGG, ...) into the s16le PCM the Parakeet engine consumes.
Useful as a pre-step for non-WAV inputs.

## JavaScript API

The high-level wrapper lives in `parakeet.js`; the addon binding
itself is in `binding.js`.

### Quick wiring

```js
const binding = require('@qvac/transcription-parakeet/binding')
const { ParakeetInterface } = require('@qvac/transcription-parakeet/parakeet')

const parakeet = new ParakeetInterface(
  binding,
  {
    modelPath: 'models/parakeet-tdt-0.6b-v3.q8_0.gguf',
    modelType: 'tdt',          // hint; auto-detected from GGUF metadata
    maxThreads: 4,
    useGPU:     false
  },
  (handle, event, jobId, output, error) => {
    if (event === 'Output' && output) {
      for (const seg of output) console.log(seg.text)
    }
  }
)

await parakeet.loadWeights({
  filename:  'parakeet-tdt-0.6b-v3.q8_0.gguf',
  chunk:     ggufBytes,                  // Uint8Array
  completed: true
})
await parakeet.activate()

await parakeet.append({ type: 'audio', data: float32AudioBuffer })
await parakeet.append({ type: 'end of job' })
// ... wait for the JobEnded event ...
await parakeet.destroyInstance()
```

The `examples/utils.js` helper module wraps the `loadWeights`
streaming + the `Output`/`JobEnded` race resolution; see
`examples/transcribe.js` for the recommended template.

### Output events

| Event       | Payload                                                        |
|-------------|----------------------------------------------------------------|
| `Output`    | `Array<Transcript>` -- `{ text, start, end, toAppend, id }` per segment |
| `JobEnded`  | nothing -- the job finished cleanly                            |
| `Error`     | error string -- non-fatal failure surfaced to JS               |

Lifecycle states: `loading` -> `listening` -> `processing` -> `listening`
-> ... -> `idle` (set on `destroyInstance`).

## Model Variants

| Variant     | Languages | Decoder                  | Default GGUF size (q8_0) | Notes |
|-------------|-----------|--------------------------|-------------------------:|-------|
| **CTC**     | English   | argmax CTC               | ~ 700 MiB                | Fast, no PnC. |
| **TDT** ⭐  | ~25       | RNN-T greedy + duration  | ~ 715 MiB                | Recommended default; PnC + auto-detect. |
| **EOU**     | English   | RNN-T greedy + `<EOU>`   | ~ 132 MiB                | Streaming-trained; lower early-utterance accuracy. |
| **Sortformer** | n/a    | Diarization head         | ~ 141 MiB                | 4-speaker, offline. |

Per-tier sizes (`f16` / `q8_0` / `q4_0`) are listed in the
[`parakeet-cpp` README](https://github.com/GustavoA1604/qvac-parakeet.cpp#supported-checkpoints).

## Development

### Running tests

```bash
npm run test:unit                                          # JS unit tests (mocked)
QVAC_TEST_GGUF_DIR=models npm run test:integration         # JS integration vs. real GGUFs
npm run test:cpp                                           # gtest C++ suite
```

The integration suite pulls each model type via
`QVAC_TEST_GGUF_DIR=<path-with-staged-ggufs>` (or per-model
overrides like `QVAC_TEST_GGUF_TDT=/full/path.gguf`). Tests skip
cleanly when no GGUF is available, so CI without local models
still passes.

### Project structure

```
qvac-lib-infer-parakeet/
├── addon/
│   ├── src/
│   │   ├── model-interface/parakeet/   # ParakeetModel.{hpp,cpp} -- thin wrapper
│   │   │                               # over qvac_parakeet::Engine
│   │   ├── js-interface/               # JSAdapter + binding.cpp
│   │   └── addon/                      # AddonCpp.hpp glue
│   └── tests/                          # gtest C++ tests
├── examples/                           # transcribe.js, diarized-transcribe.js,
│                                       # live-mic.js, live-mic-diarized.js,
│                                       # decode-audio.js, utils.js
├── lib/                                # error / logger helpers
├── scripts/                            # download-models.sh, convert-nemo.sh, ...
├── test/
│   ├── unit/                           # brittle JS unit tests
│   └── integration/                    # end-to-end JS tests
├── parakeet.js                         # JS wrapper (state machine + buffering)
├── index.js                            # TranscriptionParakeet umbrella
├── binding.js                          # native addon entry point
├── package.json
├── vcpkg.json                          # depends on `parakeet-cpp >= 2026-04-28`
└── vcpkg-configuration.json            # points at GustavoA1604/qvac-registry-vcpkg
```

## Supported Platforms

| Platform | Architecture | Status      | GGML backend |
|----------|--------------|-------------|--------------|
| macOS    | arm64, x64   | Tier 1      | Metal (default) |
| iOS      | arm64        | Tier 1      | Metal (default) |
| Linux    | arm64, x64   | Tier 1      | Vulkan (default) |
| Android  | arm64        | Tier 1      | OpenCL / Vulkan |
| Windows  | x64          | Tier 1      | Vulkan (default) |

GPU backends are selected at port-install time via the
`parakeet-cpp[metal|vulkan|opencl]` features. The `parakeet-cpp`
port bundles ggml at upstream `58c38058` so the binding is ABI-isolated
from sibling speech-stack ports (`whisper-cpp`, `chatterbox-cpp`,
`stable-diffusion-cpp`) that use the chatterbox-patched `ggml` overlay.

## License

This project is licensed under Apache-2.0 -- see [LICENSE](LICENSE) for
details. Model files are distributed under the **NVIDIA Open Model
License**; see the upstream HuggingFace cards for the per-checkpoint
terms.

## Acknowledgments

- **NVIDIA** for the Parakeet ASR + Sortformer model family.
- **The ggml team** at ggml-org for the inference runtime.
- **Tether** for the QVAC addon framework.

---

*Issues and contributions welcome on the
[qvac](https://github.com/tetherto/qvac) monorepo.*
