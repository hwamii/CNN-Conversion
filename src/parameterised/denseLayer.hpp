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
    // load data
    float dIn_tmp[W_EXP_RNUM];
    float wTmp[W_B_CNUM];
    for(int i=0; i < W_EXP_RNUM; i++){
        dIn_tmp[i] = d_in[i];
    }

    for (size_t i = 0; i < W_B_CNUM; i++) {
        for(int j=0; j < W_EXP_RNUM; j++){
            wTmp[j] = weights1D[i * W_EXP_RNUM+j];
        }

        //TODO: do the same trick for bias.
        
        // Compute dot product of input and i-th column of weights
        float sum = dotProduct<W_EXP_RNUM>(dIn_tmp, wTmp);
        d_out[i] = ReLU(sum + bias[i]);
    
        // TODO: Load data from temp variable to d_out
        
    }
}
// w0 w1 w2 w3 w4
// w0 w16 w32 w48 w64 
// Print the output values
template <size_t W_B_CNUM>
void printOutput(float d_out[W_B_CNUM]) {
    for (size_t i = 0; i < W_B_CNUM; i++) {
        std::cout << "d_out[" << i << "] = " << d_out[i] << std::endl;
    }
}