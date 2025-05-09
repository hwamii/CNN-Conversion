#pragma once

#include "denseLayer.hpp"  // that contains the dense layer implementation with the alternative activation functions
#include "loadFunction.hpp"  // that contains the loadFunction and saveToCSV functions
template <
    std::size_t D0_IN,
    std::size_t D0_OUT,
    std::size_t D1_OUT,
    std::size_t D2_OUT,
    std::size_t D3_OUT
>
void spectrumSensing(
    float input[D0_IN],

    float weights0[D0_IN * D0_OUT],
    float bias0[D0_OUT],

    float weights1[D0_OUT * D1_OUT],
    float bias1[D1_OUT],

    float weights2[D1_OUT * D2_OUT],
    float bias2[D2_OUT],

    float weights3[D2_OUT * D3_OUT],
    float bias3[D3_OUT],

    float output[D3_OUT]
) {
    // Internal layer outputs
    float out0[D0_OUT];
    float out1[D1_OUT];
    float out2[D2_OUT];
    float out3[D3_OUT];

    // Changed denseLayer function to make more compact
    dense<D0_IN, D0_OUT, RELU>(input, weights0, bias0, out0);
    dense<D0_OUT, D1_OUT, RELU>(out0, weights1, bias1, out1);
    dense<D1_OUT, D2_OUT, RELU>(out1, weights2, bias2, out2);
    dense<D2_OUT, D3_OUT, SOFTMAX>(out2, weights3, bias3, output);  // final
}