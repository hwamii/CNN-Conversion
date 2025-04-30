#pragma once

#include <iostream>
#include <iomanip>
#include <cstddef> // For size_t
#include <cmath>

enum ActivationType { NONE, RELU, SOFTMAX };

template <size_t IN, size_t OUT, ActivationType ACT = RELU>
void dense(float input[IN], float weights[IN * OUT], float bias[OUT], float output[OUT]) {
    float temp[OUT];

    for (size_t i = 0; i < OUT; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < IN; j++) {
            sum += input[j] * weights[j * OUT + i];
        }
        temp[i] = sum + bias[i];
    }

    if constexpr (ACT == RELU) {
        for (size_t i = 0; i < OUT; i++) {
            output[i] = std::max(0.0f, temp[i]);
        }
    } else if constexpr (ACT == SOFTMAX) {
        float maxVal = temp[0];
        for (size_t i = 1; i < OUT; i++) {
            if (temp[i] > maxVal) maxVal = temp[i];
        }

        float sum = 0.0f;
        for (size_t i = 0; i < OUT; i++) {
            output[i] = std::exp(temp[i] - maxVal);
            sum += output[i];
        }

        for (size_t i = 0; i < OUT; i++) {
            output[i] /= sum;
        }
    } else {
        for (size_t i = 0; i < OUT; i++) {
            output[i] = temp[i];
        }
    }
}
