#include <iostream>
#include "loadFunction.h"
#include <iomanip>
#include "denseLayer.hpp"

// Constants for array sizes
constexpr size_t WEIGHT_COLS = 512;
constexpr size_t WEIGHT_ROWS = 16;
constexpr size_t BIAS_ROWS = 512;
constexpr size_t EXPECTED_DATA = 16;
constexpr size_t NEURON_NUM = 512;

// ReLU activation function
// float ReLU(float x) {
//     return (x > 0) ? x : 0;
// }

// Compute dot product of two arrays

//Remove dotproduct function from the cpp file 
// float dotProduct(const float a[], const float b[], size_t size) {
//     float result;
//     for (size_t i = 0; i < size; i++) {
//         result += a[i] * b[i];
//     }
//     return result;
// }
