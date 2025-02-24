#include <iostream>
#include <cstdlib>
#include "weights.h"

// Define input and output sizes
#define INPUT_SIZE 16
#define OUTPUT_SIZE 16

class DenseLayer {
public:
    // Constructor loads the weights and biases from CSV files.
    DenseLayer(const std::string &weights_file, const std::string &biases_file);
    void setInput(const float new_input[INPUT_SIZE]);
    void forward();
    void printOutput();

private:
    float input[INPUT_SIZE];
    float weights[OUTPUT_SIZE][INPUT_SIZE];
    float bias[OUTPUT_SIZE];
    float output[OUTPUT_SIZE];

    float ReLU(float x);
    float dotProduct(const float a[], const float b[], int size);
};

DenseLayer::DenseLayer(const std::string &weights_file, const std::string &biases_file) {
    // Load weights using the function from weights.h
    std::vector<std::vector<float> > loadedWeights = loadWeights(weights_file, OUTPUT_SIZE, INPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            weights[i][j] = loadedWeights[i][j];
        }
    }
    
    // Load biases using the function from weights.h
    std::vector<float> loadedBiases = loadBias(biases_file, OUTPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        bias[i] = loadedBiases[i];
        output[i] = 0.0f;
    }
}

void DenseLayer::setInput(const float new_input[INPUT_SIZE]) {
    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = new_input[i];
    }
}

float DenseLayer::ReLU(float x) {
    return (x > 0) ? x : 0;
}

float DenseLayer::dotProduct(const float a[], const float b[], int size) {
    float result = 0.0f;
    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

void DenseLayer::forward() {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float sum = dotProduct(input, weights[i], INPUT_SIZE);
        output[i] = ReLU(sum + bias[i]);
    }
}

void DenseLayer::printOutput() {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        std::cout << "Neuron " << i << ": " << output[i] << std::endl;
    }
}

int main() {
    // Update these paths to match your CSV files
    std::string weightsFile = "/Users/kateaizpuru/Documents/CNN/test-data/weights-altered.csv";
    std::string biasFile = "/Users/kateaizpuru/Documents/CNN/test-data/biases.csv";
    
    DenseLayer layer(weightsFile, biasFile);
    
    // Prepare an example input array.
    float example_input[INPUT_SIZE] = {0.023172325f, 0.954666768f, 0.537868863f, 0.428133923f, 
        0.874992976f, 0.52329852f, 0.499172047f, 0.6028312f, 0.095627101f, 0.38898065f, 
        0.799446854f, 0.618940573f, 0.078196824f, 0.882892741f, 0.844261063f, 0.523200747f};
    
    layer.setInput(example_input);
    layer.forward();
    layer.printOutput();
    
    return 0;
}