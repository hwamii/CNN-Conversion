#include <iostream>
#include <vector>
#include "loadData.h" // Ensure this provides loadCSV function

#define RAW_VALUE 16
#define RESULT_SIZE 512
class DenseLayer {
public:
    DenseLayer();
    void setInput(const float new_input[RAW_VALUE]);
    void forward(const std::vector<std::vector<float> >& weights, const std::vector<float>& bias);
    void printOutput();

private:
    float dotProduct(const float a[], const std::vector<float>& b, int size);
    float input[RAW_VALUE];
    std::vector<float> output;

    float ReLU(float x);
};

// Constructor: Initializes output vector
DenseLayer::DenseLayer() : output(RESULT_SIZE, 0.0f) {}

// Set input values
void DenseLayer::setInput(const float new_input[RAW_VALUE]) {
    for (int i = 0; i < RAW_VALUE; i++) {
        input[i] = new_input[i];
    }
}

// ReLU activation function
float DenseLayer::ReLU(float x) {
    return (x > 0) ? x : 0;
}

// Compute dot product of input array and a column of the weights matrix
// May need to integrate this to the forward pass mechanism as well
float DenseLayer::dotProduct(const float a[], const std::vector<float>& b, int size) {
    float result = 0.0f;
    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Forward pass: Matrix-vector multiplication + bias + ReLU
void DenseLayer::forward(const std::vector<std::vector<float> >& weights, const std::vector<float>& bias) {
    if (weights.size() != RAW_VALUE || weights[0].size() != RESULT_SIZE || bias.size() != RESULT_SIZE) {
        std::cerr << "Error: Weights or bias dimensions are incorrect." << std::endl;
        return;
    }

    for (int i = 0; i < RESULT_SIZE; i++) {
        // Extract the i-th column of the weights matrix
        std::vector<float> weightsColumn(RAW_VALUE);
        for (int j = 0; j < RAW_VALUE; j++) {
            weightsColumn[j] = weights[j][i];
        }

        // Compute dot product of input and weights column
        float sum = dotProduct(input, weightsColumn, RAW_VALUE);
        output[i] = ReLU(sum + bias[i]);
    }
}

// Print the output values
void DenseLayer::printOutput() {
    for (int i = 0; i < RESULT_SIZE; i++) {
        std::cout << "Neuron " << i << ": " << output[i] << std::endl;
    }
}

int main() {
    DenseLayer layer;

    float example_input[RAW_VALUE] = {0.023172325, 0.954666768, 0.537868863, 0.428133923, 0.874992976, 0.52329852, 
                                       0.499172047, 0.6028312, 0.095627101, 0.38898065, 0.799446854, 0.618940573, 
                                       0.078196824, 0.882892741, 0.844261063, 0.523200747};

    layer.setInput(example_input);

    // Load weights and biases
    std::vector<std::vector<float> > weights = loadWeights("/Users/kateaizpuru/Documents/CNN/test-data/weights.csv", RAW_VALUE, RESULT_SIZE);
    std::vector<float> bias = loadBias("/Users/kateaizpuru/Documents/CNN/test-data/biases.csv", RESULT_SIZE); 

    std::cout << "Weights size: " << weights.size() << " x " << (weights.empty() ? 0 : weights[0].size()) << std::endl;
    std::cout << "Bias size: " << bias.size() << std::endl;

    // Forward pass
    layer.forward(weights, bias);

    // Print results
    layer.printOutput();

    return 0;
}