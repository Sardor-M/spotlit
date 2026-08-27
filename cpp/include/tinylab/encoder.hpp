#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace tinylab {

/// Encoder contract. Must match python/tinylab/encoder.py exactly; see ADR-0005.
struct EncoderSpec {
    int version = 1;
    std::size_t num_buckets = 4096;
    std::vector<std::size_t> ngram_sizes = {3, 4, 5};
    std::size_t max_input_bytes = 512;
};

/// One hashed n-gram bucket and its signed count.
struct Feature {
    std::uint32_t bucket = 0;
    std::int32_t count = 0;
};

[[nodiscard]] const EncoderSpec& spec_v1() noexcept;

/// UTF-8 bytes, ASCII-lowercased, non-alphanumeric collapsed to single spaces.
[[nodiscard]] std::string normalize(std::string_view text, const EncoderSpec& spec);

/// FNV-1a 64 over the n-gram, with its length mixed in.
[[nodiscard]] std::uint64_t hash_ngram(const char* gram, std::size_t size) noexcept;

/// Bucket-sorted signed counts. Integers only, so parity with Python is exact.
[[nodiscard]] std::vector<Feature> encode(std::string_view text, const EncoderSpec& spec);

}  // namespace tinylab
