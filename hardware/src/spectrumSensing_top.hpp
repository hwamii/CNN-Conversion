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
#pragma HLS INTERFACE m_axi     port=input     offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi     port=weights0  offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi     port=bias0     offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi     port=weights1  offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi     port=bias1     offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi     port=weights2  offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi     port=bias2     offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi     port=weights3  offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi     port=bias3     offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi     port=output    offset=slave bundle=gmem

#pragma HLS INTERFACE s_axilite port=input     bundle=control
#pragma HLS INTERFACE s_axilite port=weights0  bundle=control
#pragma HLS INTERFACE s_axilite port=bias0     bundle=control
#pragma HLS INTERFACE s_axilite port=weights1  bundle=control
#pragma HLS INTERFACE s_axilite port=bias1     bundle=control
#pragma HLS INTERFACE s_axilite port=weights2  bundle=control
#pragma HLS INTERFACE s_axilite port=bias2     bundle=control
#pragma HLS INTERFACE s_axilite port=weights3  bundle=control
#pragma HLS INTERFACE s_axilite port=bias3     bundle=control
#pragma HLS INTERFACE s_axilite port=output    bundle=control
#pragma HLS INTERFACE s_axilite port=return    bundle=control

    spectrumSensing<16, 512, 256, 64, 2>(
        input,
        weights0, bias0,
        weights1, bias1,
        weights2, bias2,
        weights3, bias3,
        output
    );
}