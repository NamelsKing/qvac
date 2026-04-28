# Quick Start Guide

Get the ggml-backend Parakeet binding (sourced from
[qvac-parakeet.cpp](https://github.com/tetherto/qvac-parakeet.cpp))
running end-to-end in
about ten minutes (most of which is downloading + converting the
upstream NeMo `.nemo` files).

## 1. Build the addon

```bash
git clone https://github.com/tetherto/qvac.git
cd qvac/packages/qvac-lib-infer-parakeet
npm install            # also runs bare-make to compile the addon
```

`npm install` pulls the `parakeet-cpp` overlay port (which bundles
ggml at the pinned upstream commit) and produces
`prebuilds/<platform>-<arch>/qvac__transcription-parakeet.bare`.

Linux only: install Clang/LLVM 19 with libc++ first, e.g.
```bash
sudo apt install clang libc++-dev libc++abi-dev build-essential pkg-config
```

## 2. Stage a model

The ggml backend takes a single `.gguf` per checkpoint. The standard
flow is "download `.nemo` from HuggingFace, convert to `.gguf` via
`qvac-parakeet.cpp`'s converter":

```bash
npm run setup-models                   # all 4 models, q8_0 (recommended)
# or
npm run setup-models -- -t tdt         # just TDT
npm run setup-models -- -t eou -q f16  # full-precision EOU
```

Output GGUFs land in `./models/`. The conversion step uses
qvac-parakeet.cpp's Python venv automatically when a sibling
`~/dev/qvac-parakeet.cpp` checkout is present; otherwise pass
`--python /path/to/venv/bin/python` (NeMo + `gguf` + numpy + torch
required).

## 3. Run an example

Examples take a model path (and audio path / `--accumulate`) -- type
auto-detected, GPU on, threads picked by the engine. Tweak knobs by
editing the example if needed.

```bash
bare examples/transcribe.js \
     --model models/parakeet-tdt-0.6b-v3.q8_0.gguf \
     --audio examples/samples/sample-16k.wav

bare examples/transcribe.js \
     --model models/sortformer-4spk-v1.q8_0.gguf \
     --audio examples/samples/two-speakers-16k.wav
```

If you prefer `npm run example -- ...`, remember the `--` separator
so npm forwards args to the script.

## 4. Combine ASR + diarization

```bash
bare examples/diarized-transcribe.js \
     --asr-model  models/parakeet-tdt-0.6b-v3.q8_0.gguf \
     --diar-model models/sortformer-4spk-v1.q8_0.gguf \
     --audio      examples/samples/two-speakers-16k.wav
```

## 5. Live microphone

```bash
bare examples/live-mic.js --model models/parakeet-eou-120m-v1.q8_0.gguf
bare examples/live-mic.js --model models/parakeet-eou-120m-v1.q8_0.gguf --accumulate

bare examples/live-mic-diarized.js \
     --asr-model  models/parakeet-tdt-0.6b-v3.q8_0.gguf \
     --diar-model models/sortformer-4spk-v1.q8_0.gguf --accumulate
```

Captures the default input device via `sox -d` (install: `brew install sox`,
`apt install sox`, `choco install sox`). With `--accumulate`,
transcripts are appended onto one line per turn and flushed on
silence, speaker change, or Ctrl-C.

## 6. Build your own

The two-script pattern from `examples/transcribe.js` is the
recommended template -- copy it, swap the GGUF + audio, and tweak
the per-segment callback. The `examples/utils.js` helpers handle
the `loadWeights` streaming + `Output`/`JobEnded` race resolution
so you don't need to. See [README.md](README.md) for the full
JavaScript API.

## Next steps

- 📖 [Full README](README.md) -- API, model variants, supported platforms.
- 💻 [examples/](examples/) -- `transcribe.js`, `diarized-transcribe.js`,
  `live-mic.js`, `live-mic-diarized.js`, `decode-audio.js`, `utils.js`.
- 🔧 [scripts/](scripts/) -- `download-models.sh`, `convert-nemo.sh`,
  `trigger-benchmark.sh`.

## Model comparison

| Model        | Languages | Decoder        | GGUF (q8_0) | Best for |
|--------------|-----------|----------------|------------:|----------|
| **TDT** ⭐    | ~25       | RNN-T + duration | ~715 MiB    | General-purpose multilingual |
| **CTC**      | English   | argmax CTC     | ~700 MiB    | Fast English, no PnC |
| **EOU**      | English   | RNN-T + `<EOU>` | ~132 MiB    | Real-time streaming, end-of-turn |
| **Sortformer** | n/a     | Diarization head | ~141 MiB    | 4-speaker diarization |

⭐ = recommended default.

## Getting help

- 🐛 [Report issues](https://github.com/tetherto/qvac/issues)
- 💬 [Discussions](https://github.com/tetherto/qvac/discussions)
