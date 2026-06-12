#include "ops.hpp"

#include <cstdlib>
#include <cstring>

#include "ggml.h"

// NOLINTBEGIN(readability-identifier-naming,readability-identifier-length)
// Op helpers use math-style parameter names (x, OC, s0..s1, p0..p1, d0..d1,
// W_target, H_target) that mirror the ggml C-API documentation.

namespace easyocr::ggml::ops {

namespace {

// Opt-in (OCR_GGML_CRAFT_BIAS_BROADCAST=1): add the channel bias via ggml_add's
// implicit broadcast instead of materialising a full [W,H,OC,N] copy with
// ggml_repeat. ggml_add officially broadcasts its second operand
// (ggml_can_repeat) and the CPU/Vulkan/Metal kernels all implement it, so the
// result is numerically identical while saving a buffer + an op per conv.
// Default off until CI confirms it across backends. Read via getenv at
// graph-build time (not a hot path) so a single-process test can toggle it.
bool bias_broadcast_enabled() {
  const char* v = std::getenv("OCR_GGML_CRAFT_BIAS_BROADCAST");
  return v != nullptr && std::strcmp(v, "1") == 0;
}

// Add a [OC] bias to a [W, H, OC, N] activation map. By default we broadcast
// via ggml_repeat (the historical, always-supported path); the env flag opts
// into ggml_add's implicit broadcast, dropping the ggml_repeat.
::ggml_tensor* add_channel_bias(
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    ::ggml_context* ctx, ::ggml_tensor* x, ::ggml_tensor* bias) {
  const int64_t oc = bias->ne[0];
  auto* b4 = ggml_reshape_4d(ctx, bias, 1, 1, oc, 1);
  if (bias_broadcast_enabled()) {
    return ggml_add(ctx, x, b4);
  }
  return ggml_add(ctx, x, ggml_repeat(ctx, b4, x));
}

} // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
::ggml_tensor* conv_2d_bias(
    ::ggml_context* ctx, ::ggml_tensor* x, ::ggml_tensor* kernel,
    ::ggml_tensor* bias, int s0, int s1, int p0, int p1, int d0, int d1) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  auto* y = ggml_conv_2d(ctx, kernel, x, s0, s1, p0, p1, d0, d1);
  return add_channel_bias(ctx, y, bias);
}

::ggml_tensor* conv_2d_bias_relu(
    ::ggml_context* ctx, ::ggml_tensor* x, ::ggml_tensor* kernel,
    ::ggml_tensor* bias, int s0, int s1, int p0, int p1, int d0, int d1) {
  return ggml_relu(
      ctx, conv_2d_bias(ctx, x, kernel, bias, s0, s1, p0, p1, d0, d1));
}

::ggml_tensor* bilinear_to(
    ::ggml_context* ctx, ::ggml_tensor* x, int64_t W_target, int64_t H_target) {
  // PyTorch's `F.interpolate(..., mode='bilinear', align_corners=False)`
  // (the CRAFT U-net default) corresponds to ggml's BILINEAR mode WITHOUT
  // the ALIGN_CORNERS flag.  ggml's flag-driven coord formula matches
  // PyTorch's `align_corners=False`:
  //     src = (dst + 0.5) * (src_size / dst_size) - 0.5
  return ggml_interpolate(
      ctx,
      x,
      W_target,
      H_target,
      x->ne[2],
      x->ne[3],
      static_cast<uint32_t>(GGML_SCALE_MODE_BILINEAR));
}

} // namespace easyocr::ggml::ops

// NOLINTEND(readability-identifier-naming,readability-identifier-length)
