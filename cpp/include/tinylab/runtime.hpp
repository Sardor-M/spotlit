#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace tinylab {

[[nodiscard]] std::string_view version() noexcept;

/// A tensor in the model manifest; `scale_offset` is meaningful only when quantized.
struct TensorInfo {
    std::size_t data_offset = 0;
    std::size_t scale_offset = 0;
    std::vector<int> shape;
    bool quantized = false;

    [[nodiscard]] std::size_t element_count() const noexcept;
};

/// Expand per-output-channel symmetric INT8 weights into float; row `c` scaled by `scales[c]`.
void dequantize_per_channel_int8(
    const std::int8_t* quantized_values,
    const float* scales,
    std::size_t output_channel_count,
    std::size_t values_per_output_channel,
    float* output_values
) noexcept;

}  // namespace tinylab
