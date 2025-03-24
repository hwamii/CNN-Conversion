#pragma once

#include <iostream>
#include <iomanip>
#include <cstddef> // For size_t

// ReLU activation function
inline float ReLU(float x) {
    return (x > 0) ? x : 0;
}

// Compute dot product of two arrays
template <size_t SIZE>
float dotProduct(const float a[SIZE], const float b[SIZE]) {
    float result = 0.0f;
    for (size_t i = 0; i < SIZE; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Forward pass: Matrix-vector multiplication + bias + ReLU
template <size_t W_EXP_RNUM, size_t W_B_CNUM>
void forward(float d_in[W_EXP_RNUM], float weights1D[W_EXP_RNUM * W_B_CNUM], float bias[W_B_CNUM], float d_out[W_B_CNUM]) {
    for (size_t i = 0; i < W_B_CNUM; i++) {
        // Compute dot product of input and i-th column of weights
        float sum = dotProduct<W_EXP_RNUM>(d_in, &weights1D[i * W_EXP_RNUM]);
        d_out[i] = ReLU(sum + bias[i]);
    }
}

// Print the output values
template <size_t W_B_CNUM>
void printOutput(float d_out[W_B_CNUM]) {
    for (size_t i = 0; i < W_B_CNUM; i++) {
        std::cout << "d_out[" << i << "] = " << d_out[i] << std::endl;
    }
}