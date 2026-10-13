#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "tinylab/encoder.hpp"

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

/// The linear student from a `.tlab` file (python/tinylab/export/README.md): per-row INT8 table.
struct LinearModel {
    EncoderSpec spec;
    std::size_t dims = 0;
    std::vector<float> scales;
    std::vector<std::int8_t> table;
};

/// Read a version 1 `.tlab` file; throws std::runtime_error on a malformed or foreign file.
[[nodiscard]] LinearModel load_tlab(const std::string& path);

/// L2-normalized embedding; cosine similarity is the dot product of two of these.
[[nodiscard]] std::vector<float> embed(const LinearModel& model, std::string_view text);

}  // namespace tinylab
