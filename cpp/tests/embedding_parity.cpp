// Asserts the C++ runtime reproduces the exporter's reference embeddings (goldens.txt).

#include "tinylab/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double kTolerance = 1e-6;

std::string from_hex(const std::string& hex) {
    if (hex == "-") {
        return {};
    }
    std::string out;
    for (std::size_t index = 0; index + 1 < hex.size(); index += 2) {
        out.push_back(static_cast<char>(std::stoul(hex.substr(index, 2), nullptr, 16)));
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: %s model.tlab goldens.txt\n", argv[0]);
        return 2;
    }
    try {
        const tinylab::LinearModel model = tinylab::load_tlab(argv[1]);
        std::ifstream file(argv[2]);
        if (!file) {
            std::printf("cannot open %s\n", argv[2]);
            return 2;
        }

        int checked = 0;
        int failed = 0;
        double worst = 0.0;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream stream(line);
            std::string keyword;
            std::string name;
            std::string hex;
            std::size_t dims = 0;
            if (!(stream >> keyword >> name >> hex >> dims) || keyword != "case") {
                continue;
            }
            std::vector<double> expected(dims);
            for (double& value : expected) {
                stream >> value;
            }
            const std::vector<float> actual = tinylab::embed(model, from_hex(hex));
            double diff = actual.size() == dims ? 0.0 : INFINITY;
            for (std::size_t d = 0; d < std::min(dims, actual.size()); ++d) {
                diff = std::max(diff, std::abs(expected[d] - actual[d]));
            }
            worst = std::max(worst, diff);
            ++checked;
            if (diff > kTolerance) {
                std::printf("  FAIL %s: max diff %.3g\n", name.c_str(), diff);
                ++failed;
            }
        }
        if (checked == 0) {
            std::printf("no cases found in %s\n", argv[2]);
            return 2;
        }
        std::printf("embedding parity: %d/%d texts match, max abs diff %.3g\n", checked - failed, checked, worst);
        return failed == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::printf("error: %s\n", error.what());
        return 1;
    }
}
