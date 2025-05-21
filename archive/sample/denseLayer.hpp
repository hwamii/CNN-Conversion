// denseLayer.hpp
#pragma once

#include <cstddef>
#ifdef __SYNTHESIS__
#include <hls_math.h>
#else
#include <cmath>
#endif

enum ActivationType { NONE, RELU, SOFTMAX };

template <typename T,
          std::size_t IN,
          std::size_t OUT,
          ActivationType ACT>
          
void dense(const T input[IN], const T weights[IN * OUT], const T bias[OUT], T output[OUT]) {
    T temp[OUT];

    for (std::size_t i = 0; i < OUT; i++) {
        T sum = 0;
        for (std::size_t j = 0; j < IN; j++) {
            sum += input[j] * weights[j * OUT + i];
        }
        temp[i] = sum + bias[i];
    }

    switch (ACT) {
        case RELU:
            for (std::size_t i = 0; i < OUT; i++) {
                output[i] = (temp[i] > 0) ? temp[i] : 0;
            }
            break;

        case SOFTMAX: {
            T maxVal = temp[0];
            for (std::size_t i = 1; i < OUT; i++) {
                if (temp[i] > maxVal) maxVal = temp[i];
            }

            T sumExp = 0;
            for (std::size_t i = 0; i < OUT; i++) {
    #ifdef __SYNTHESIS__
                T exp_val = (T)hls::exp((float)(temp[i] - maxVal));
    #else
                T exp_val = (T)std::exp((float)(temp[i] - maxVal));
    #endif
                output[i] = exp_val;
                sumExp += exp_val;
            }

            for (std::size_t i = 0; i < OUT; i++) {
                output[i] = output[i] / sumExp;
            }
            break;
        }

        case NONE:
        default:
            for (std::size_t i = 0; i < OUT; i++) {
                output[i] = temp[i];
            }
            break;
    }
}

