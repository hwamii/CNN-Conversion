#pragma once

#include <iostream>
#include <iomanip>
#include <cstddef> // For size_t
#include <cmath>

enum ActivationType { NONE, RELU, SOFTMAX };

template <typename T, 
        size_t IN, 
        size_t OUT, 
        ActivationType ACT = RELU
        >

void dense(const T input[IN], const T weights[IN * OUT], const T bias[OUT], T output[OUT]) {
    T temp[OUT];

    for (size_t i = 0; i < OUT; i++) {
        T sum = (T)0;
        for (size_t j = 0; j < IN; j++) {
            sum += input[j] * weights[j * OUT + i];
        }
        temp[i] = sum + bias[i];
    }

    if constexpr (ACT == RELU) {
        for (size_t i = 0; i < OUT; i++) {
            output[i] = (temp[i] > (T)0) ? temp[i] : (T)0;
        }
    } else if constexpr (ACT == SOFTMAX) {
        T maxVal = temp[0];
        for (size_t i = 1; i < OUT; i++) {
            if (temp[i] > maxVal) maxVal = temp[i];
        }

        T sumExp= (T)0;
        for (size_t i = 0; i < OUT; i++) {
            output[i] = (sizeof(T)==4
                ? expf(temp[i] - maxVal)
                : exp(temp[i] - maxVal));
            sumExp += output[i];
        }

        for (size_t i = 0; i < OUT; i++) {
            output[i] /= sumExp;
        }
    } else {
        for (size_t i = 0; i < OUT; i++) {
            output[i] = temp[i];
        }
    }
}
