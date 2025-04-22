#include <iostream>
#include "loadFunction.hpp"
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
    ModelPaths paths;

    // Layer 0
    float input[D0_IN];
    float weights0[D0_IN * D0_OUT], bias0[D0_OUT];
    float out0[D0_OUT];

    loadFunction(paths.dense0.input, 1, D0_IN, input, false);
    loadFunction(paths.dense0.weights, D0_IN, D0_OUT, weights0, false);
    loadFunction(paths.dense0.bias, 1, D0_OUT, bias0, true);
    dense<D0_IN, D0_OUT, RELU>(input, weights0, bias0, out0);
    saveToCSV(paths.dense0.output, out0, D0_OUT);

    // Layer 1
    float weights1[D0_OUT * D1_OUT], bias1[D1_OUT], out1[D1_OUT];
    loadFunction(paths.dense1.weights, D0_OUT, D1_OUT, weights1, false);
    loadFunction(paths.dense1.bias, 1, D1_OUT, bias1, true);
    dense<D0_OUT, D1_OUT, RELU>(out0, weights1, bias1, out1);
    saveToCSV(paths.dense1.output, out1, D1_OUT);

    // Layer 2
    float weights2[D1_OUT * D2_OUT], bias2[D2_OUT], out2[D2_OUT];
    loadFunction(paths.dense2.weights, D1_OUT, D2_OUT, weights2, false);
    loadFunction(paths.dense2.bias, 1, D2_OUT, bias2, true);
    dense<D1_OUT, D2_OUT, RELU>(out1, weights2, bias2, out2);
    saveToCSV(paths.dense2.output, out2, D2_OUT);

    // Layer 3 (final with softmax)
    float weights3[D2_OUT * D3_OUT], bias3[D3_OUT], out3[D3_OUT];
    loadFunction(paths.dense3.weights, D2_OUT, D3_OUT, weights3, false);
    loadFunction(paths.dense3.bias, 1, D3_OUT, bias3, true);
    dense<D2_OUT, D3_OUT, SOFTMAX>(out2, weights3, bias3, out3);
    saveToCSV(paths.dense3.output, out3, D3_OUT);

    std::cout << "Final output (softmax):" << std::endl;
    std::cout << "Final prediction:" << std::endl;
        for (std::size_t i = 0; i < D3_OUT; ++i) {
            std::cout << "Class " << i << ": " << out3[i] << std::endl;
        }
}
// int main() {
//     // Instantiate paths
//     ModelPaths paths;

//     // Buffers
//     float input[D0_IN];

//     float weights0[D0_IN * D0_OUT], bias0[D0_OUT];
//     float weights1[D0_OUT * D1_OUT], bias1[D1_OUT];
//     float weights2[D1_OUT * D2_OUT], bias2[D2_OUT];
//     float weights3[D2_OUT * D3_OUT], bias3[D3_OUT];

//     float output[D3_OUT];

//     // ===== Load inputs for each layer =====
//     loadFunction(paths.dense0.input, 1, D0_IN, input, false);
//     loadFunction(paths.dense0.weights, D0_IN, D0_OUT, weights0, false);
//     loadFunction(paths.dense0.bias, 1, D0_OUT, bias0, true);

//     loadFunction(paths.dense1.weights, D0_OUT, D1_OUT, weights1, false);
//     loadFunction(paths.dense1.bias, 1, D1_OUT, bias1, true);

//     loadFunction(paths.dense2.weights, D1_OUT, D2_OUT, weights2, false);
//     loadFunction(paths.dense2.bias, 1, D2_OUT, bias2, true);

//     loadFunction(paths.dense3.weights, D2_OUT, D3_OUT, weights3, false);
//     loadFunction(paths.dense3.bias, 1, D3_OUT, bias3, true);

//     // ===== Run the neural network =====
//     neural_net_top<
//         D0_IN, D0_OUT, D1_OUT, D2_OUT, D3_OUT
//     >(
//         input,
//         weights0, bias0,
//         weights1, bias1,
//         weights2, bias2,
//         weights3, bias3,
//         output
//     );

//     // ===== Save output =====
//     saveToCSV(paths.dense3.output, output, D3_OUT);

//     std::cout << "Final prediction:" << std::endl;
//     for (std::size_t i = 0; i < D3_OUT; ++i) {
//         std::cout << "Class " << i << ": " << output[i] << std::endl;
//     }
//     return 0;
// }