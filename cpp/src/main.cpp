#include "tinylab/runtime.hpp"

#include <cstdio>

/// Placeholder entry point. Exists so the toolchain is proven before there is a model.
int main() {
    std::printf("tinylab runtime %s\n", tinylab::version().data());
    std::printf("No model loaded. See README.md for the roadmap.\n");
    return 0;
}
