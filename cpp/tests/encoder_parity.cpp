// Asserts the C++ encoder reproduces tests/encoder_goldens.txt byte for byte.

#include "tinylab/encoder.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string from_hex(const std::string& hex) {
    if (hex == "-") {
        return {};
    }
    std::string out;
    out.reserve(hex.size() / 2);
    for (std::size_t index = 0; index + 1 < hex.size(); index += 2) {
        out.push_back(static_cast<char>(std::stoul(hex.substr(index, 2), nullptr, 16)));
    }
    return out;
}

struct Case {
    std::string name;
    std::string text;
    std::vector<tinylab::Feature> expected;
};

bool parse_case(const std::string& line, Case& parsed) {
    std::istringstream stream(line);
    std::string keyword;
    std::string hex;
    std::size_t count = 0;
    if (!(stream >> keyword >> parsed.name >> hex >> count) || keyword != "case") {
        return false;
    }
    parsed.text = from_hex(hex);

    std::string pair;
    while (stream >> pair) {
        const std::size_t colon = pair.find(':');
        if (colon == std::string::npos) {
            return false;
        }
        parsed.expected.push_back(
            {static_cast<std::uint32_t>(std::stoul(pair.substr(0, colon))),
             static_cast<std::int32_t>(std::stol(pair.substr(colon + 1)))}
        );
    }
    return parsed.expected.size() == count;
}

void report(const Case& parsed, const std::vector<tinylab::Feature>& actual) {
    std::printf("  FAIL %s: expected %zu features, got %zu\n", parsed.name.c_str(),
                parsed.expected.size(), actual.size());
    const std::size_t limit = std::min(parsed.expected.size(), actual.size());
    for (std::size_t index = 0; index < limit; ++index) {
        if (parsed.expected[index].bucket != actual[index].bucket ||
            parsed.expected[index].count != actual[index].count) {
            std::printf("       first difference at %zu: golden %u:%d, got %u:%d\n", index,
                        parsed.expected[index].bucket, parsed.expected[index].count,
                        actual[index].bucket, actual[index].count);
            return;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "tests/encoder_goldens.txt";
    std::ifstream file(path);
    if (!file) {
        std::printf("cannot open %s\n", path.c_str());
        return 2;
    }

    int checked = 0;
    int failed = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind("case ", 0) != 0) {
            continue;
        }
        Case parsed;
        if (!parse_case(line, parsed)) {
            std::printf("  FAIL malformed golden line: %.60s\n", line.c_str());
            ++failed;
            continue;
        }

        const std::vector<tinylab::Feature> actual =
            tinylab::encode(parsed.text, tinylab::spec_v1());
        ++checked;

        bool equal = actual.size() == parsed.expected.size();
        for (std::size_t index = 0; equal && index < actual.size(); ++index) {
            equal = actual[index].bucket == parsed.expected[index].bucket &&
                    actual[index].count == parsed.expected[index].count;
        }
        if (!equal) {
            report(parsed, actual);
            ++failed;
        }
    }

    if (checked == 0) {
        std::printf("no cases found in %s\n", path.c_str());
        return 2;
    }
    std::printf("encoder parity: %d/%d cases match\n", checked - failed, checked);
    return failed == 0 ? 0 : 1;
}
