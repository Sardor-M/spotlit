#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace tinylab {

/// Semantic version of the runtime, reported by the CLI.
[[nodiscard]] std::string_view version() noexcept;

/// A tensor as it appears in the model manifest.
///
/// Mirrors the layout Moka uses: weights are per-output-channel symmetric INT8 with a
/// float32 scale per output channel; biases stay float32. `scale_offset` is meaningful
/// only when `quantized` is true.
///
/// See docs/learnings/00-reference-analysis/moka/04-deployment-quantization-runtime.md
struct TensorInfo {
    std::size_t data_offset = 0;
    std::size_t scale_offset = 0;
    std::vector<int> shape;
    bool quantized = false;

    [[nodiscard]] std::size_t element_count() const noexcept;
};

/// Expand per-output-channel symmetric INT8 weights into float.
///
/// `output_channel_count` rows of `values_per_output_channel` weights each, where row `c`
/// is scaled by `scales[c]`. Dequantizing once at load trades memory for a small download;
/// integer inference (needed on a microcontroller) is a different implementation.
void dequantize_per_channel_int8(
    const std::int8_t* quantized_values,
    const float* scales,
    std::size_t output_channel_count,
    std::size_t values_per_output_channel,
    float* output_values
) noexcept;

}  // namespace tinylab
