#include <iostream>
#include "loadFunction.h"

#include <iomanip>

// Constants for array sizes
#define WEIGHT_COLS 512
#define WEIGHT_ROWS 16
#define BIAS_ROWS 512
#define BIAS_COLS 1
#define EXPECTED_DATA 16
#define NEURON_NUM 512

class DenseLayer {
public:
    DenseLayer();
    void forward(float d_in[EXPECTED_DATA], float* weights1D, float bias[BIAS_ROWS], float d_out[NEURON_NUM]);
    void printOutput(const float d_out[NEURON_NUM]);

private:
    float dotProduct(const float a[], const float b[], int size);
    float ReLU(float x);
};

// Constructor
DenseLayer::DenseLayer() {
    // No initialization needed for now
}

// ReLU activation function
float DenseLayer::ReLU(float x) {
    return (x > 0) ? x : 0;
}

// Compute dot product of two arrays
float DenseLayer::dotProduct(const float a[], const float b[], int size) {
    float result = 0.0f;
    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Forward pass: Matrix-vector multiplication + bias + ReLU
void DenseLayer::forward(float d_in[EXPECTED_DATA], float* weights1D, float bias[BIAS_ROWS], float d_out[NEURON_NUM]) {
    // Unwrap the 1D weights array into a 2D array
    float weights[WEIGHT_ROWS][WEIGHT_COLS];
    for (int i = 0; i < WEIGHT_ROWS; i++) {
        for (int j = 0; j < WEIGHT_COLS; j++) {
            weights[i][j] = weights1D[i * WEIGHT_COLS + j];
        }
    }

    for (int i = 0; i < NEURON_NUM; i++) {
        // Extract the i-th column of the weights matrix
        float weightsColumn[WEIGHT_ROWS];
        for (int j = 0; j < WEIGHT_ROWS; j++) {
            weightsColumn[j] = weights[j][i];
        }

        // Compute dot product of input and weights column
        float sum = dotProduct(d_in, weightsColumn, WEIGHT_ROWS);
        d_out[i] = ReLU(sum + bias[i]);
        // std::cout << std::fixed << std::setprecision(8) << d_out[i] << std::endl;
    }
}

// Print the output values
void DenseLayer::printOutput(const float d_out[NEURON_NUM]) {
    for (int i = 0; i < NEURON_NUM; i++) {
        std::cout << d_out[i] << std::endl;
    }
}
int main() {
    // Load weights and bias
    std::string weightsFile = "/Users/kateaizpuru/Documents/CNN/test-data/weights.csv"; // Replace with your weights file name
    float* weights1D = loadFunction(weightsFile, WEIGHT_ROWS, WEIGHT_COLS, false);

    std::string biasFile = "/Users/kateaizpuru/Documents/CNN/test-data/biases.csv"; // Replace with your bias file name
    float* bias1D = loadFunction(biasFile, BIAS_ROWS, BIAS_COLS, true);

    // Create a DenseLayer object
    DenseLayer denseLayer;

    // Set input values
    float example_input[EXPECTED_DATA] = {
        0.023172325, 0.954666768, 0.537868863, 0.428133923, 0.874992976, 0.52329852,
        0.499172047, 0.6028312, 0.095627101, 0.38898065, 0.799446854, 0.618940573,
        0.078196824, 0.882892741, 0.844261063, 0.523200747
    };

    // Perform forward pass
    float d_out[NEURON_NUM]; // Creates an empty array to store the output
    denseLayer.forward(example_input, weights1D, bias1D, d_out);

    // Print the output values
    denseLayer.printOutput(d_out);

    

    // Free the 1D arrays
    freeData(weights1D);
    freeData(bias1D);

    return 0;
}