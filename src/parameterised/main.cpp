#include "loadFunction.h"
#include "denseLayer.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstddef>
#include <fstream>
// Define constants for array sizes

// Dense0
constexpr std::size_t D_INPUT = 16;   // Number of rows in weights matrix (input size)
constexpr std::size_t D_OUTPUT = 512; // Number of columns in weights matrix (output size)

constexpr std::size_t D_WROWS = 16;
constexpr std::size_t D_BWCOLS = 512;

// Dense1
constexpr std::size_t D1_INPUT = 512;  // Number of rows in weights matrix (input size)
constexpr std::size_t D1_OUTPUT = 256; // Number of columns in weights matrix (output size)

constexpr std::size_t D1_WROWS = 512;
constexpr std::size_t D1_BWCOLS = 256;

// Dense2
constexpr std::size_t D2_INPUT = 256;  // Number of rows in weights matrix (input size)
constexpr std::size_t D2_OUTPUT = 64; // Number of columns in weights matrix (output size)
constexpr std::size_t D2_WROWS = 256;
constexpr std::size_t D2_BWCOLS = 64;

// Dense3
constexpr std::size_t D3_INPUT = 64;  // Number of rows in weights matrix (input size)
constexpr std::size_t D3_OUTPUT = 2; // Number of columns in weights matrix (output size)
constexpr std::size_t D3_WROWS = 64;
constexpr std::size_t D3_BWCOLS = 2;


void saveToCSV(const std::string &filename, const float *data, std::size_t length) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    for (std::size_t i = 0; i < length; i++) {
        file << data[i];
        if (i < length - 1) {
            file << ",";
        }
    }
    file.close();
}

int main()
{
    //Flattened data for Dense, read from CSV
    float denseInput[D_INPUT]; // Input is 16 (from flatten.csv)
    float denseWeights[D_INPUT * D_OUTPUT]; // Weights are 16x512 (from dense_weights.csv)
    float denseBias[D_OUTPUT]; // Bias is 512 (from dense_biases.csv)

    //Read data from CSV files
    loadFunction("/Users/kateaizpuru/Documents/CNN/src/layerData/flatten.csv", 1, D_INPUT, denseInput, false); // 512
    loadFunction("/Users/kateaizpuru/Documents/CNN/src/layerData/dense/dense_weights.csv", D_INPUT, D_OUTPUT, denseWeights, false);
    loadFunction("/Users/kateaizpuru/Documents/CNN/src/layerData/dense/dense_biases.csv", 1, D_OUTPUT, denseBias, true);
    
    //Forward pass for Dense Layer
    float dense0Output[D_OUTPUT]; // Output is 512 (from dense_output.csv)
    dense<D_INPUT, D_OUTPUT>(denseInput, denseWeights, denseBias, dense0Output);
    // Print the output values
    // std::cout << "Dense Layer Output:" << std::endl;
    // printOutput<D_OUTPUT>(dense0Output);

    //Save the output to a CSV file
    saveToCSV("/Users/kateaizpuru/Documents/CNN/src/layerData/dense/dense_output.csv", dense0Output, D_OUTPUT);
    
    

    // Data for Dense1, read from CSV
    float dense1Input[D1_INPUT]; // Input is 512 (from dense_output.csv)
    float dense1Weights[D1_INPUT * D1_OUTPUT]; // Weights are 512x256 (from dense1_weights.csv)
    float dense1Bias[D1_OUTPUT]; // Bias is 256 (from dense1_biases.csv)
    //Read data from CSV files
    loadFunction("/Users/kateaizpuru/Documents/CNN/src/layerData/dense/dense.csv", 1, D1_INPUT, dense1Input, false); // 512
    // Weights should be 512 (columns) x 256 (rows)
    loadFunction("/Users/kateaizpuru/Documents/CNN/src/layerData/dense1/dense_1_weights.csv", D1_INPUT, D1_OUTPUT, dense1Weights, false);
    loadFunction("/Users/kateaizpuru/Documents/CNN/src/layerData/dense1/dense_1_biases.csv", 1, D1_OUTPUT, dense1Bias, true);
    
    //Forward pass for Dense1 Layer
    float dense1Output[D1_OUTPUT]; // Output is 256 (from dense1_output.csv)
    dense<D1_INPUT, D1_OUTPUT>(dense1Input, dense1Weights, dense1Bias, dense1Output);
  
    //Save the output to a CSV file
    saveToCSV("/Users/kateaizpuru/Documents/CNN/src/layerData/dense1/dense1_output.csv", dense1Output, D1_OUTPUT);


    // Dense Layer 2
    float dense2Input[D1_OUTPUT]; // Input is 256 (from dense1_output.csv)
    float dense2Weights[D1_OUTPUT * D2_OUTPUT]; // Weights are 256x64 (from dense2_weights.csv)
    float dense2Bias[D2_OUTPUT]; // Bias is 64 (from dense2_biases.csv)

    //Read data from CSV files
    loadFunction("/Users/kateaizpuru/Documents/CNN/src/layerData/dense1/dense1_output.csv", 1, D1_OUTPUT, dense2Input, false);
    loadFunction("/Users/kateaizpuru/Documents/CNN/src/layerData/dense2/dense_2_weights.csv", D1_OUTPUT, D2_OUTPUT, dense2Weights, false);
    loadFunction("/Users/kateaizpuru/Documents/CNN/src/layerData/dense2/dense_2_biases.csv", 1, D2_OUTPUT, dense2Bias, true);
    float dense2Output[D2_OUTPUT]; // Output is 64 (from dense2_output.csv)
    dense<D1_OUTPUT, D2_OUTPUT>(dense2Input, dense2Weights, dense2Bias, dense2Output);
    // Print the output values

    // std::cout << "Dense Layer 2 Output:" << std::endl;
    // printOutput<D2_OUTPUT>(dense2Output);
    //Save the output to a CSV file
    saveToCSV("/Users/kateaizpuru/Documents/CNN/src/layerData/dense2/dense2_output.csv", dense2Output, D2_OUTPUT);

    // Dense Layer 3
    float dense3Input[D2_OUTPUT]; // Input is 64 (from dense2_output.csv)
    float dense3Weights[D2_OUTPUT * D3_OUTPUT]; // Weights are 64x2 (from dense3_weights.csv)
    float dense3Bias[D3_OUTPUT]; // Bias is 2 (from dense3_biases.csv)
    //Read data from CSV files
    loadFunction("/Users/kateaizpuru/Documents/CNN/src/layerData/dense2/dense2_output.csv", 1, 64, dense3Input, false);
    loadFunction("/Users/kateaizpuru/Documents/CNN/src/layerData/dense3/dense_3_weights.csv", 64, 2, dense3Weights, false);
    loadFunction("/Users/kateaizpuru/Documents/CNN/src/layerData/dense3/dense_3_biases.csv", 1, 2, dense3Bias, true);
    //Forward pass for Dense3 Layer
    float dense3Output[D3_OUTPUT]; // Output is 2 (from dense3_output.csv)
    float dense3Prob[D3_OUTPUT]; // Output is 2 (from dense3_output.csv)
    denseFinal<D2_OUTPUT, D3_OUTPUT>(dense3Input, dense3Weights, dense3Bias, dense3Output);

    // Print the output values
    softmax<D3_OUTPUT>(dense3Output, dense3Prob);
    printOutput<D3_OUTPUT>(dense3Prob);
    //Save the output to a CSV file
    saveToCSV("/Users/kateaizpuru/Documents/CNN/src/layerData/dense3/dense3_output.csv", dense3Prob, D3_OUTPUT);
}


