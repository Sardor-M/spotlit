// Command-line runtime: embed texts with a .tlab model, time each, and compare them.

#include "tinylab/runtime.hpp"

#include <chrono>
#include <cstdio>
#include <exception>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("tinylab runtime %s\nusage: %s model.tlab TEXT [TEXT ...]\n", tinylab::version().data(),
                    argv[0]);
        return argc == 1 ? 0 : 2;
    }
    try {
        const tinylab::LinearModel model = tinylab::load_tlab(argv[1]);
        std::printf("model: %zu buckets x %zu dims\n", model.spec.num_buckets, model.dims);

        constexpr int rounds = 1000;
        std::vector<std::vector<float>> vectors;
        for (int index = 2; index < argc; ++index) {
            std::vector<float> vector;
            const auto start = std::chrono::steady_clock::now();
            for (int round = 0; round < rounds; ++round) {
                vector = tinylab::embed(model, argv[index]);
            }
            const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;
            std::printf("%8.4f ms  [%+.4f %+.4f %+.4f ...]  %s\n", elapsed.count() / rounds, vector[0], vector[1],
                        vector[2], argv[index]);
            vectors.push_back(std::move(vector));
        }
        for (std::size_t other = 1; other < vectors.size(); ++other) {
            double dot = 0.0;
            for (std::size_t d = 0; d < model.dims; ++d) {
                dot += static_cast<double>(vectors[0][d]) * vectors[other][d];
            }
            std::printf("cosine(text 1, text %zu) = %.4f\n", other + 1, dot);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
