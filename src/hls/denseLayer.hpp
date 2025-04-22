#pragma once

#include <iostream>
#include <iomanip>
#include <cstddef> // For size_t
#include <cmath>

// ReLU activation function
inline float ReLU(float x) {
    return (x > 0) ? x : 0;
}

template <size_t SIZE>
void softmax(float input[SIZE], float output[SIZE]) {
    float maxVal = input[0];
    for (size_t i = 1; i < SIZE; ++i) {
        if (input[i] > maxVal) maxVal = input[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < SIZE; ++i) {
        output[i] = std::exp(input[i] - maxVal); // for numerical stability
        sum += output[i];
    }

    for (size_t i = 0; i < SIZE; ++i) {
        output[i] /= sum;
    }
}

// Gives similar results to the above function
// template <size_t W_ROWS, size_t W_COLS>
// void dense(float d_in[W_ROWS], float weights[W_ROWS * W_COLS], float bias[W_COLS], float d_out[W_COLS]) {
//     for (size_t out_neuron = 0; out_neuron < W_COLS; out_neuron++) {  // For each output neuron (0-15)
//         float sum = 0.0f;
//         for (size_t in_feat = 0; in_feat < W_ROWS; in_feat++) {  // For each input feature (0-512)
//             sum += d_in[in_feat] * weights[in_feat * W_COLS + out_neuron];  // Compute dot product
//         }
//         d_out[out_neuron] = ReLU(sum + bias[out_neuron]);  // Apply ReLU activation
//     }
// }

template <size_t IN, size_t OUT>
void dense(float d_in[IN], float weights[IN * OUT], float bias[OUT], float d_out[OUT]) {
    for (size_t i = 0; i < OUT; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < IN; j++) {
            sum += d_in[j] * weights[j * OUT + i];
        }
        d_out[i] = std::max(0.0f, sum + bias[i]);  // ReLU inline
    }
}

template <size_t W_ROWS, size_t W_COLS>
void denseFinal(float d_in[W_ROWS], float weights[W_ROWS * W_COLS], float bias[W_COLS], float d_out[W_COLS]) {
    for (size_t out_neuron = 0; out_neuron < W_COLS; out_neuron++) {  // For each output neuron (0-15)
        float sum = 0.0f;
        for (size_t in_feat = 0; in_feat < W_ROWS; in_feat++) {  // For each input feature (0-512)
            sum += d_in[in_feat] * weights[in_feat * W_COLS + out_neuron];  // Compute dot product
        }
        d_out[out_neuron] = sum + bias[out_neuron];  // Apply ReLU activation
    }
}

// Print the output values
template <size_t W_COLS>
void printOutput(float d_out[W_COLS]) {
    for (size_t i = 0; i < W_COLS; i++) {
        std::cout << d_out[i] << std::endl;
    }
}