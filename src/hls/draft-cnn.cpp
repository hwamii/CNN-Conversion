#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

#define RAW_VALUE 16 //Input file has 16 values
#define OUTPUT_SIZE 512 //Output file (from Teams) has 512 values

// The weights array is 16x512, where each row contains the weights for a single neuron. 
#define WEIGHTS_ROW 16 //Each row has 16 values (the actual file has 512 values per row)
#define WEIGHTS_COL 512 //There are 512 cols

class DenseLayer {
public:
    // The constructor loads the weights from the CSV file.
    DenseLayer(const std::string &weights_file, const std::string &biases_file);
    void setInput(const float new_input[RAW_VALUE]);
    void forward();
    void printOutput();
    void forwardfeed(const float input[RAW_VALUE], const float weights[RAW_VALUE][OUTPUT_SIZE], const float bias[OUTPUT_SIZE]);

private:
    float input[RAW_VALUE];
    float weights[RAW_VALUE][OUTPUT_SIZE];
    float bias[OUTPUT_SIZE];  
    float output[OUTPUT_SIZE];

    float ReLU(float x);
    float dotProduct(const float a[], const float b[], int size);
    void loadWeights(const std::string &filename);
    void loadBiases(const std::string &filename);
};

// Load weights from CSV file into the weights array.
void DenseLayer::loadWeights(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open weights file: " << filename << std::endl;
        exit(1);
    }

    std::string line;

    for (int i = 0; i < WEIGHTS_ROW; i++) { //I'm assuming 16 since the input file has 16 rows
        if (!std::getline(file, line)) {
            std::cerr << "Error: Not enough lines in weights file." << std::endl;
            exit(1);
        }

        std::cout << "Reading line " << i << ": " << line << std::endl; // Debug print

        std::stringstream ss(line);
        std::string value;
        int j = 0;
        while (std::getline(ss, value, ',')) {
            try {
                weights[i][j] = std::stof(value);
            } catch (const std::invalid_argument &e) {
                std::cerr << "Error: Invalid number in weights file at row " << i 
                          << ", column " << j << " (value: '" << value << "')" << std::endl;
                exit(1);
            }
            j++;
            if (j > WEIGHTS_COL) { //Expected 512 values per row
                std::cerr << "Error: Too many values in row " << i << " of weights file." << std::endl;
                exit(1);
            }
        }
    }
    std::cout << "Weights loaded successfully." << std::endl;
    file.close();
}

void DenseLayer::loadBiases(const std::string &filename) {
    std::ifstream file(filename);
    std::string line;
    std::string value;
    for (int i = 0; i < OUTPUT_SIZE; i++) { //Expected output size is 512 (from the Teams file, not the Remote Desktop file)
        if (!std::getline(file, line)) {
            std::cerr << "Error: Not enough lines in bias file." << std::endl;
            exit(1);
            
        }
        std::stringstream ss(line);
        ss >> bias[i];
        std::cout << "value " << i << bias[i] << std::endl;

        // understand how the bias file was read
        // have a standard load/read file function
        // take the stream and read it into the bias array
        // pointer and address of array that the data is stored in
        // size (cols and rows) -- need to establish array size
        // install gdb and remove the cout stuff
    }
    file.close();
    std::cout << "Biases loaded successfully." << std::endl;
}

// Constructor: loads the weights and initializes biases and outputs.
// !! Will need to modularise to have it on a separate file
DenseLayer::DenseLayer(const std::string &weights_file, const std::string &biases_file) {
    loadWeights(weights_file);
    loadBiases(biases_file);
}

// Set input values from an external array.
void DenseLayer::setInput(const float new_input[RAW_VALUE]) {
    for (int i = 0; i < RAW_VALUE; i++) {
        input[i] = new_input[i];
    }
}

// ReLU activation function.
float DenseLayer::ReLU(float x) {
    return (x > 0) ? x : 0;
}

// Compute dot product of two arrays.
float DenseLayer::dotProduct(const float a[], const float b[], int size) {
    float result = 0.0f;
    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Forward pass: computes each neuron's output.
void DenseLayer::forward() {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float weightsRow[16];
        for (int j = 0; j < RAW_VALUE; j++) {
            weightsRow[j] = weights[j][i];
        }
        float sum = dotProduct(input, weightsRow, RAW_VALUE); //need to extract row and put it in an array
        output[i] = ReLU(sum + bias[i]);
        // float sum = dotProduct(input, weights[i], RAW_VALUE); //need to extract row and put it in an array
        // output[i] = ReLU(sum + bias[i]);
    }
}

void DenseLayer::forwardfeed(const float input[RAW_VALUE], const float weights[RAW_VALUE][OUTPUT_SIZE], const float bias[OUTPUT_SIZE]) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float weightsRow[16];
        for (int j = 0; j < RAW_VALUE; j++) {
            weightsRow[j] = weights[j][i];
        }
        float sum = dotProduct(input, weightsRow, RAW_VALUE); //need to extract row and put it in an array
        output[i] = ReLU(sum + bias[i]);
        // float sum = dotProduct(input, weights[i], RAW_VALUE); //need to extract row and put it in an array
        // output[i] = ReLU(sum + bias[i]);
    }
}
// Print the outputs for each neuron.
void DenseLayer::printOutput() {
    for (int i = 1; i <= OUTPUT_SIZE; i++) {
        // std::cout << "Neuron " << i << ": " << output[i] << std::endl;
        std::cout << output[i] << std::endl;

    }
}


int main() {
    // The CSV file should contain 16 rows, each with 512 values.
    DenseLayer layer("/Users/kateaizpuru/Documents/CNN/test-data/weights.csv", "/Users/kateaizpuru/Documents/CNN/test-data/biases.csv");
    std::cout << "Layer created." << std::endl;
    float example_input[RAW_VALUE] = {0.023172325, 0.954666768, 0.537868863, 0.428133923, 0.874992976, 0.52329852, 0.499172047, 0.6028312, 
        0.095627101, 0.38898065, 0.799446854, 0.618940573, 0.078196824, 0.882892741, 0.844261063, 0.523200747};


    layer.setInput(example_input);
    std::cout << "Input set." << std::endl;
    layer.forward();
    std::cout << "Forward pass completed." << std::endl;
    layer.printOutput();


    return 0;
}


// subtract the expected output from the actual output and if not 0, then its incorrect
// forward function needs to be fixed
// should have parameters passed into it forward(input, weights, biases)