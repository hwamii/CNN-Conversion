#include "spectrumSensing.hpp"

void spectrumSensing_top(
    float input[16],
    float weights0[16 * 512],
    float bias0[512],
    float weights1[512 * 256],
    float bias1[256],
    float weights2[256 * 64],
    float bias2[64],
    float weights3[64 * 2],
    float bias3[2],
    float output[2]
) {



#pragma PIPELINE off
    spectrumSensing<16, 512, 256, 64, 2>(
        input,
        weights0, bias0,
        weights1, bias1,
        weights2, bias2,
        weights3, bias3,
        output
    );
}