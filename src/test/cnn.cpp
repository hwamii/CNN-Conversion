#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "loadData.h" // Ensure this provides loadCSV function

//expected denseLayer.forward takes parameters (float[512][16] weights, float[512] bias, int expected_inputs, int expected_outputs)
//expected denseLayer.setInput takes parameters (float[16] input)
//expected denseLayer.printOutput takes no parameters


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
    void forward(const float weights[weightRows][weightCols], const float bias[biasRows], int expected_inputs, int expected_outputs);
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
void DenseLayer::forward(const float weights[weightRows][weightCols], const float bias[biasRows], int expected_inputs, int expected_outputs) {
    if (expected_inputs != weightRows || expected_outputs != biasRows) {
        std::cerr << "Error: Weights or bias dimensions are incorrect." << std::endl;
        return;
    }

    for (int i = 0; i < neurons; i++) {
        // Extract the i-th column of the weights matrix
        float weightsColumn[weightRows];
        for (int j = 0; j < weightRows; j++) {
            weightsColumn[j] = weights[j][i];
        }

        // Compute dot product of input and weights column
        float sum = dotProduct(input, weightsColumn, weightRows);
        output[i] = ReLU(sum + bias[i]);
    }
}

// Print the output values
void DenseLayer::printOutput() {
    for (int i = 0; i < neurons; i++) {
        std::cout << "Neuron " << i << ": " << output[i] << std::endl;
    }
}

int main() {
    // Load weights and bias
    std::string weightsFile = "/Users/kateaizpuru/Documents/CNN/test-data/weights.csv"; 
    float** weightsData = loadData(weightsFile, weightRows, weightCols, false);

    std::string biasFile = "/Users/kateaizpuru/Documents/CNN/test-data/biases.csv"; 
    float** biasData = loadData(biasFile, biasRows, biasCols, true);

    // Create a DenseLayer object
    DenseLayer denseLayer;

    // Set input values
    float example_input[expectedData] = {
        0.023172325, 0.954666768, 0.537868863, 0.428133923, 0.874992976, 0.52329852,
        0.499172047, 0.6028312, 0.095627101, 0.38898065, 0.799446854, 0.618940573,
        0.078196824, 0.882892741, 0.844261063, 0.523200747
    };
    denseLayer.setInput(example_input);

    // Convert weights and bias to static arrays
    float weights[weightRows][weightCols];
    for (int i = 0; i < weightRows; i++) {
        for (int j = 0; j < weightCols; j++) {
            weights[i][j] = weightsData[i][j];
        }
    }

    float bias[biasRows];
    for (int i = 0; i < biasRows; i++) {
        bias[i] = biasData[i][0];
    }

    // Free dynamically allocated memory
    freeData(weightsData, weightRows);
    freeData(biasData, biasRows);

    // Perform forward pass
    denseLayer.forward(weights, bias, weightRows, biasRows);

    // Print the output values
    denseLayer.printOutput();

    return 0;
}