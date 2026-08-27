#include "tinylab/encoder.hpp"

namespace tinylab {
namespace {

constexpr std::uint64_t kFnvOffsetBasis = 0xcbf29ce484222325ULL;
constexpr std::uint64_t kFnvPrime = 0x100000001b3ULL;
constexpr unsigned char kSpace = 0x20;

}  // namespace

const EncoderSpec& spec_v1() noexcept {
    static const EncoderSpec spec{};
    return spec;
}

std::string normalize(std::string_view text, const EncoderSpec& spec) {
    std::string out;
    out.reserve(text.size());

    bool previous_was_space = true;
    for (const char raw : text) {
        const auto byte = static_cast<unsigned char>(raw);
        unsigned char current = kSpace;
        if (byte >= 0x41 && byte <= 0x5a) {
            current = static_cast<unsigned char>(byte + 0x20);
        } else if ((byte >= 0x61 && byte <= 0x7a) || (byte >= 0x30 && byte <= 0x39) ||
                   byte >= 0x80) {
            current = byte;
        }

        if (current == kSpace) {
            if (previous_was_space) {
                continue;
            }
            previous_was_space = true;
        } else {
            previous_was_space = false;
        }
        out.push_back(static_cast<char>(current));
    }

    if (!out.empty() && static_cast<unsigned char>(out.back()) == kSpace) {
        out.pop_back();
    }
    if (out.size() > spec.max_input_bytes) {
        out.resize(spec.max_input_bytes);
    }
    return out;
}

std::uint64_t hash_ngram(const char* gram, std::size_t size) noexcept {
    std::uint64_t digest = kFnvOffsetBasis;
    for (std::size_t index = 0; index < size; ++index) {
        digest ^= static_cast<unsigned char>(gram[index]);
        digest *= kFnvPrime;
    }
    digest ^= static_cast<std::uint64_t>(size);
    digest *= kFnvPrime;
    return digest;
}

std::vector<Feature> encode(std::string_view text, const EncoderSpec& spec) {
    const std::string normalized = normalize(text, spec);
    if (normalized.empty()) {
        return {};
    }

    std::string padded;
    padded.reserve(normalized.size() + 2);
    padded.push_back(static_cast<char>(kSpace));
    padded.append(normalized);
    padded.push_back(static_cast<char>(kSpace));

    std::vector<std::int32_t> counts(spec.num_buckets, 0);
    for (const std::size_t size : spec.ngram_sizes) {
        if (padded.size() < size) {
            continue;
        }
        const std::size_t last_start = padded.size() - size;
        for (std::size_t start = 0; start <= last_start; ++start) {
            const std::uint64_t digest = hash_ngram(padded.data() + start, size);
            const std::size_t bucket = digest % spec.num_buckets;
            counts[bucket] += (digest >> 63) ? -1 : 1;
        }
    }

    std::vector<Feature> features;
    for (std::size_t bucket = 0; bucket < counts.size(); ++bucket) {
        if (counts[bucket] != 0) {
            features.push_back({static_cast<std::uint32_t>(bucket), counts[bucket]});
        }
    }
    return features;
}

}  // namespace tinylab
