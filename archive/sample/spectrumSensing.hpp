#pragma once

#include "denseLayer.hpp"
#include <cstddef>

template <
    typename T,
    std::size_t D0_IN,
    std::size_t D0_OUT,
    std::size_t D1_OUT,
    std::size_t D2_OUT,
    std::size_t D3_OUT
>
void spectrumSensing(
    T input[D0_IN],

    T weights0[D0_IN * D0_OUT],
    T bias0[D0_OUT],

    T weights1[D0_OUT * D1_OUT],
    T bias1[D1_OUT],

    T weights2[D1_OUT * D2_OUT],
    T bias2[D2_OUT],

    T weights3[D2_OUT * D3_OUT],
    T bias3[D3_OUT],

    T output[D3_OUT]
) {
    T out0[D0_OUT];
    T out1[D1_OUT];
    T out2[D2_OUT];

#pragma HLS PIPELINE off

    dense<float, D0_IN,  D0_OUT,  RELU>    (input,   weights0, bias0, out0);
    dense<float, D0_OUT, D1_OUT,  RELU>    (out0,    weights1, bias1, out1);
    dense<float, D1_OUT, D2_OUT,  RELU>    (out1,    weights2, bias2, out2);
    dense<float, D2_OUT, D3_OUT, SOFTMAX>  (out2,    weights3, bias3, output);
}
 
