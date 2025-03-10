#include <iostream>
#include "loadFunction.h"

// Constants for array sizes
#define weightCols 512
#define weightRows 16
#define biasRows 512
#define biasCols 1
#define expectedData 16
#define neurons 512

class DenseLayer {
public:
    DenseLayer();
    void setInput(const float new_input[expectedData]);
    void forward(const float* weights1D, const float bias[biasRows], int expected_inputs, int expected_outputs);
    void printOutput();

private:
    float dotProduct(const float a[], const float b[], int size);
    float input[expectedData];
    float output[neurons];
    float ReLU(float x);
};

// Constructor
DenseLayer::DenseLayer() {
    for (int i = 0; i < neurons; i++) {
        output[i] = 0.0f;
    }
}

// Set input values
void DenseLayer::setInput(const float new_input[expectedData]) {
    for (int i = 0; i < expectedData; i++) {
        input[i] = new_input[i];
    }
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
void DenseLayer::forward(float d_in[16], const float* weights1D, const float bias[biasRows], int expected_inputs, float d_out[512]) { //Write output in a 1d array
    if (expected_inputs != weightRows || expected_outputs != biasRows) {
        std::cerr << "Error: Weights or bias dimensions are incorrect." << std::endl;
        return;
    }

    // Unwrap the 1D weights array into a 2D array
    float weights[weightRows][weightCols];
    for (int i = 0; i < weightRows; i++) {
        for (int j = 0; j < weightCols; j++) {
            weights[i][j] = weights1D[i * weightCols + j];
        }
    }

    for (int i = 0; i < neurons; i++) {
        // Extract the i-th column of the weights matrix
        float weightsColumn[weightRows];
        for (int j = 0; j < weightRows; j++) {
            weightsColumn[j] = weights[j][i];
        }

        // Compute dot product of input and weights column
        float sum = dotProduct(d_in, weightsColumn, weightRows);
        d_out[i] = ReLU(sum + bias[i]);
    }
}

// Print the output values
void DenseLayer::printOutput() {
    for (int i = 0; i < neurons; i++) {
        std::cout << output[i] << std::endl;
    }
}

int main() {
    // Load weights and bias
    std::string weightsFile = "/Users/kateaizpuru/Documents/CNN/test-data/weights.csv"; // Replace with your weights file name
    float* weights1D = loadFunction(weightsFile, weightRows, weightCols, false);

    std::string biasFile = "/Users/kateaizpuru/Documents/CNN/test-data/biases.csv"; // Replace with your bias file name
    float* bias1D = loadFunction(biasFile, biasRows, biasCols, true);

    // Create a DenseLayer object
    DenseLayer denseLayer;

    // Set input values
    float example_input[expectedData] = {
        0.023172325, 0.954666768, 0.537868863, 0.428133923, 0.874992976, 0.52329852,
        0.499172047, 0.6028312, 0.095627101, 0.38898065, 0.799446854, 0.618940573,
        0.078196824, 0.882892741, 0.844261063, 0.523200747
    };
    denseLayer.setInput(example_input);

    // Perform forward pass
    float d_out[neurons];
    denseLayer.forward(example_input, weights1D, bias1D, weightRows, biasRows, d_out);

    // Print the output values
    denseLayer.printOutput();

    // Free the 1D arrays
    freeData(weights1D);
    freeData(bias1D);

    return 0;
}