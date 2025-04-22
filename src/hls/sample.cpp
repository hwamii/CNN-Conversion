#include <iostream>
#include "loadFunction.h"
#include "denseLayer.hpp"
#include "cnn_top.hpp"
#include "paths.hpp"

// Constants
constexpr std::size_t D0_IN = 16;
constexpr std::size_t D0_OUT = 512;
constexpr std::size_t D1_OUT = 256;
constexpr std::size_t D2_OUT = 64;
constexpr std::size_t D3_OUT = 2;

int main() {
    // Instantiate paths
    ModelPaths paths;

    // Buffers
    float input[D0_IN];

    float weights0[D0_IN * D0_OUT], bias0[D0_OUT];
    float weights1[D0_OUT * D1_OUT], bias1[D1_OUT];
    float weights2[D1_OUT * D2_OUT], bias2[D2_OUT];
    float weights3[D2_OUT * D3_OUT], bias3[D3_OUT];

    float output[D3_OUT];

    // ===== Load inputs for each layer =====
    loadFunction(paths.dense0.input, 1, D0_IN, input, false);
    loadFunction(paths.dense0.weights, D0_IN, D0_OUT, weights0, false);
    loadFunction(paths.dense0.bias, 1, D0_OUT, bias0, true);

    loadFunction(paths.dense1.weights, D0_OUT, D1_OUT, weights1, false);
    loadFunction(paths.dense1.bias, 1, D1_OUT, bias1, true);

    loadFunction(paths.dense2.weights, D1_OUT, D2_OUT, weights2, false);
    loadFunction(paths.dense2.bias, 1, D2_OUT, bias2, true);

    loadFunction(paths.dense3.weights, D2_OUT, D3_OUT, weights3, false);
    loadFunction(paths.dense3.bias, 1, D3_OUT, bias3, true);

    // ===== Run the neural network =====
    neural_net_top<
        D0_IN, D0_OUT, D1_OUT, D2_OUT, D3_OUT
    >(
        input,
        weights0, bias0,
        weights1, bias1,
        weights2, bias2,
        weights3, bias3,
        output
    );

    // ===== Save output =====
    saveToCSV(paths.dense3.output, output, D3_OUT);

    // Optional: print output
    std::cout << "Final prediction:" << std::endl;
    printOutput<D3_OUT>(output);

    return 0;
}