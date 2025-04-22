#pragma once

#include <iostream>
#include <iomanip>
#include <cstddef> // For size_t

// ReLU activation function
inline float ReLU(float x) {
    return (x > 0) ? x : 0;
}

template <size_t SIZE>
float dotProduct(const float a[SIZE], const float b[SIZE]) {
    float result = 0.0f;
    for (size_t i = 0; i < SIZE; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Gives similar results to the above function
template <size_t W_ROWS, size_t W_COLS>
void forward(float d_in[W_ROWS], float weights[W_ROWS * W_COLS], float bias[W_COLS], float d_out[W_COLS]) {
    for (size_t out_neuron = 0; out_neuron < W_COLS; out_neuron++) {  // For each output neuron (0-15)
        float sum = 0.0f;
        for (size_t in_feat = 0; in_feat < W_ROWS; in_feat++) {  // For each input feature (0-512)
            sum += d_in[in_feat] * weights[in_feat * W_COLS + out_neuron];  // Compute dot product
        }
        d_out[out_neuron] = ReLU(sum + bias[out_neuron]);  // Apply ReLU activation
    }
}

// w0 w1 w2 w3 w4
// w0 w16 w32 w48 w64 
// Print the output values
template <size_t W_COLS>
void printOutput(float d_out[W_COLS]) {
    for (size_t i = 0; i < W_COLS; i++) {
        std::cout << d_out[i] << std::endl;
        // std::cout << "d_out[" << i << "] = " <<  d_out[i] << std::endl;
    }
}