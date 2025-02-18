#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>  // For random initialization, may be removed

#define INPUT_SIZE 16 //As per the input size from the csv file
#define OUTPUT_SIZE 16 //As per the output size from the csv file

class DenseLayer
{
public:
    DenseLayer(const std::string &weights_file);
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
    void loadWeights(const std::string &filename);
    // void loadBias(const std::string &filename);
};

/* Load weights.csv into the program and store it in the weights array [NOT SURE HOW THIS SHOULD WORK-- 
csv still being adjusted to be read properly] */
void DenseLayer::loadWeights(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open weights file: " << filename << std::endl;
        exit(1);
    }

    std::string line;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (!std::getline(file, line)) {
            std::cerr << "Error: Not enough lines in weights file." << std::endl;
            exit(1);
        }

        std::cout << "Reading line " << i << ": " << line << std::endl;  // Debug print

        std::stringstream ss(line);
        std::string value;
        int j = 0;

        while (std::getline(ss, value, ',')) {
            weights[i][j] = std::stof(value);
            j++;
        }
    file.close();
    }
}

// Constructor with weights file as parameter - works (looks like error is with reading) - works
DenseLayer::DenseLayer(const std::string &weights_file) {
    loadWeights(weights_file);
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = 0.0f;
    }
}

// Set input values, from input array (manually set by me) - works
void DenseLayer::setInput(const float new_input[INPUT_SIZE]) {
    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = new_input[i];
    }
}

// ReLU Activation Function - works
float DenseLayer::ReLU(float x) {
    return (x > 0) ? x : 0;
}

// dotProduct Function - works
float DenseLayer::dotProduct(const float a[], const float b[], int size) {
    float result = 0.0f;
    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Forward Function - not tested (still trying to understand better)
void DenseLayer::forward() {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float sum = dotProduct(input, weights[i], INPUT_SIZE);
        output[i] = ReLU(sum + bias[i]);
    }
}

// Printg Output Function - works (only tried with fake inputs and random weights/biases)
void DenseLayer::printOutput() {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        std::cout << "Neuron " << i << ": " << output[i] << std::endl;
    }
}

int main() {
    DenseLayer layer("weights.csv");

    // Input from inputs.csv (16 values) - real values from data file
    float example_input[INPUT_SIZE] = {0.023172325, 0.954666768, 0.537868863, 0.428133923, 0.874992976, 0.52329852, 0.499172047, 0.6028312, 
                                       0.095627101, 0.38898065, 0.799446854, 0.618940573, 0.078196824, 0.882892741, 0.844261063, 0.523200747};

    layer.setInput(example_input);
    layer.forward();
    layer.printOutput();

    return 0;
}