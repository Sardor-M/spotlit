#include "tinylab/runtime.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <numeric>
#include <stdexcept>

static_assert(std::endian::native == std::endian::little, ".tlab files are little-endian");

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

namespace {

std::uint32_t read_u32(const std::vector<char>& blob, std::size_t offset) {
    if (offset + 4 > blob.size()) {
        throw std::runtime_error("truncated .tlab header");
    }
    std::uint32_t value = 0;
    std::memcpy(&value, blob.data() + offset, 4);
    return value;
}

}  // namespace

LinearModel load_tlab(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }
    const std::vector<char> blob{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    if (blob.size() < 28 || std::memcmp(blob.data(), "TLAB", 4) != 0 || read_u32(blob, 4) != 1) {
        throw std::runtime_error(path + " is not a version 1 .tlab file");
    }

    LinearModel model;
    model.spec.version = static_cast<int>(read_u32(blob, 8));
    model.spec.num_buckets = read_u32(blob, 12);
    model.dims = read_u32(blob, 16);
    model.spec.max_input_bytes = read_u32(blob, 20);
    const std::size_t size_count = read_u32(blob, 24);
    model.spec.ngram_sizes.clear();
    for (std::size_t index = 0; index < size_count; ++index) {
        model.spec.ngram_sizes.push_back(read_u32(blob, 28 + 4 * index));
    }

    const std::size_t scales_start = 28 + 4 * size_count;
    const std::size_t table_start = scales_start + 4 * model.spec.num_buckets;
    const std::size_t table_bytes = model.spec.num_buckets * model.dims;
    if (blob.size() != table_start + table_bytes) {
        throw std::runtime_error(path + " has the wrong size for its header");
    }
    model.scales.resize(model.spec.num_buckets);
    std::memcpy(model.scales.data(), blob.data() + scales_start, 4 * model.spec.num_buckets);
    model.table.resize(table_bytes);
    std::memcpy(model.table.data(), blob.data() + table_start, table_bytes);
    return model;
}

std::vector<float> embed(const LinearModel& model, std::string_view text) {
    std::vector<double> sum(model.dims, 0.0);
    for (const Feature& feature : encode(text, model.spec)) {
        const double weight = static_cast<double>(feature.count) * model.scales[feature.bucket];
        const std::int8_t* row = model.table.data() + feature.bucket * model.dims;
        for (std::size_t d = 0; d < model.dims; ++d) {
            sum[d] += weight * row[d];
        }
    }
    double norm = 0.0;
    for (const double value : sum) {
        norm += value * value;
    }
    norm = std::sqrt(norm);

    std::vector<float> out(model.dims, 0.0F);
    if (norm > 0.0) {
        for (std::size_t d = 0; d < model.dims; ++d) {
            out[d] = static_cast<float>(sum[d] / norm);
        }
    }
    return out;
}

}  // namespace tinylab
