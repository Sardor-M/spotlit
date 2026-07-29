#include "tinylab/runtime.hpp"

#include <functional>
#include <numeric>

namespace tinylab {

std::string_view version() noexcept {
    return "0.0.0";
}

std::size_t TensorInfo::element_count() const noexcept {
    return std::accumulate(
        shape.begin(), shape.end(), std::size_t{1}, std::multiplies<std::size_t>{}
    );
}

void dequantize_per_channel_int8(
    const std::int8_t* quantized_values,
    const float* scales,
    std::size_t output_channel_count,
    std::size_t values_per_output_channel,
    float* output_values
) noexcept {
    for (std::size_t channel = 0; channel < output_channel_count; ++channel) {
        const float scale = scales[channel];
        const std::size_t offset = channel * values_per_output_channel;

        for (std::size_t index = 0; index < values_per_output_channel; ++index) {
            output_values[offset + index] =
                static_cast<float>(quantized_values[offset + index]) * scale;
        }
    }
}

}  // namespace tinylab
