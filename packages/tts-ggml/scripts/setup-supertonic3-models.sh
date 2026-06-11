#!/usr/bin/env bash
#
# setup-supertonic3-models.sh — provision Supertonic 3 GGUFs locally.
#
# Downloads the public Supertonic 3 model from Hugging Face
# (Supertone/supertonic-3), converts it to a reference f32 GGUF, and then
# requantizes it to the block-quant tiers the integration tests exercise
# (q8_0 / q4_0 by default).  Everything is produced *locally* — no S3 / QVAC
# registry access required — so CI can validate the quantized v3 models
# before the artifacts are published to the bucket.  Once the GGUFs land on
# S3 this script is superseded by `npm run download-models:registry`.
#
# The Supertonic converter is ONNX-based, so the Python deps are light
# (huggingface_hub + onnx + onnxruntime + gguf + numpy — no torch).
#
# Output (under <models-dir>, default ./models):
#   supertonic3-q8_0.gguf   (~126 MB, near-transparent)
#   supertonic3-q4_0.gguf   (~80 MB, usable / size-constrained)
#   supertonic3-f16.gguf    (~191 MB, only when 'f16' requested)
#   supertonic3-f32.gguf    (~398 MB reference, only when 'f32' requested)
#
# These names match the `supertonic3-<quant>.gguf` files that
# test/utils/downloadModel.js::ensureSupertonic3Model resolves.
#
# It is idempotent: an already-present GGUF of the right size is reused, and
# the HF snapshot + f32 base are cached, so re-runs are cheap.
#
# Usage:
#   scripts/setup-supertonic3-models.sh [--models-dir DIR] [--quants "q8_0 q4_0"]
#                                       [--work-dir DIR] [--force]
#
# Env overrides mirror the flags: MODELS_DIR, QUANTS, WORK_DIR.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$HERE/.." && pwd)"

REPO_ID="Supertone/supertonic-3"
MODELS_DIR="${MODELS_DIR:-$PKG_DIR/models}"
# q8_0 + q4_0 are the tiers the quant integration test sweeps.  Add f16 / f32
# here if you also want those staged locally.
QUANTS="${QUANTS:-q8_0 q4_0}"
WORK_DIR="${WORK_DIR:-${TMPDIR:-/tmp}/supertonic3-setup}"
FORCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
        --quants)     QUANTS="$2"; shift 2 ;;
        --work-dir)   WORK_DIR="$2"; shift 2 ;;
        --force)      FORCE=1; shift ;;
        -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

HF_DIR="$WORK_DIR/hf"
VENV="$WORK_DIR/.venv"
BASE_F32="$WORK_DIR/supertonic3-f32.gguf"

mkdir -p "$MODELS_DIR" "$HF_DIR"

echo "==> Supertonic 3 model provisioning (local, no S3)"
echo "    repo-id    : $REPO_ID"
echo "    models-dir : $MODELS_DIR"
echo "    quants     : $QUANTS"
echo "    work-dir   : $WORK_DIR"

# --- 1. Python env -----------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
    echo "==> creating venv at $VENV"
    python3 -m venv "$VENV"
fi
PY="$VENV/bin/python"
echo "==> ensuring python deps (huggingface_hub onnx onnxruntime gguf numpy)"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet huggingface_hub onnx onnxruntime gguf numpy

# --- 2. Download + convert to the f32 reference GGUF ------------------------
# The f32 GGUF is the requantizer source (requantize-gguf.py dequantizes the
# source to f32 internally, so quantizing from f32 is the validated path used
# by the S3 bundle builder).
if [ "$FORCE" = "1" ] || [ ! -f "$BASE_F32" ]; then
    echo "==> converting $REPO_ID -> $BASE_F32 (ftype=f32)"
    # No --validate here: the synthesis integration test is the functional
    # check, and skipping the ONNX-Runtime parity pass keeps CI lean + reduces
    # the failure surface on runners without spare memory.
    "$PY" "$HERE/convert-supertonic2-to-gguf.py" \
        --arch supertonic3 \
        --repo-id "$REPO_ID" \
        --download-dir "$HF_DIR" \
        --out "$BASE_F32" \
        --ftype f32
else
    echo "==> reusing cached f32 base GGUF: $BASE_F32"
fi

# --- 3. Produce each requested quant tier -----------------------------------
# requantize-gguf.py keeps the Supertonic raw-read roster (voices / CFG null
# tokens / style+expand constants / embeddings / duration weights) at F32 and
# squeezes the ConvNeXt pointwise (1x1) convs to 2-D so ggml can block-quantize
# them; the tts-cpp loader re-expands them via supertonic.pwconv_squeezed.
for q in $QUANTS; do
    out="$MODELS_DIR/supertonic3-$q.gguf"
    if [ "$FORCE" != "1" ] && [ -f "$out" ]; then
        echo "==> reusing cached $out"
        continue
    fi
    case "$q" in
        f32)
            echo "==> staging f32 reference -> supertonic3-f32.gguf"
            cp "$BASE_F32" "$out"
            ;;
        f16)
            echo "==> converting f16 -> supertonic3-f16.gguf"
            "$PY" "$HERE/convert-supertonic2-to-gguf.py" \
                --arch supertonic3 \
                --repo-id "$REPO_ID" \
                --download-dir "$HF_DIR" \
                --out "$out" \
                --ftype f16
            ;;
        *)
            echo "==> quantizing -> supertonic3-$q.gguf"
            "$PY" "$HERE/requantize-gguf.py" "$BASE_F32" "$out" "$q"
            ;;
    esac
done

# --- 4. Summary --------------------------------------------------------------
echo ""
echo "==> done. staged under $MODELS_DIR:"
for q in $QUANTS; do
    f="$MODELS_DIR/supertonic3-$q.gguf"
    [ -f "$f" ] && printf "    %-26s %s\n" "supertonic3-$q.gguf" "$(du -h "$f" | awk '{print $1}')"
done
