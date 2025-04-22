#include "loadFunction.h"
#include "denseLayer.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstddef>
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

int main()
{
    // DENSE 0

    // File paths
    std::string dense0Weights = "/Users/kateaizpuru/Documents/CNN/test-data/dense0/weights.csv"; // Replace with your weights file path
    std::string dense0Bias = "/Users/kateaizpuru/Documents/CNN/test-data/dense0/biases.csv";     // Replace with your bias file path

    float d0Weights[D_INPUT * D_OUTPUT];
    loadFunction(dense0Weights, D_INPUT, D_OUTPUT, d0Weights, false);

    float d0Bias[D_BWCOLS];
    loadFunction(dense0Bias, 1, 512, d0Bias, true);
    // Input array
    // float d0Input[D_INPUT] = {
    //     0.023172325, 0.954666768, 0.537868863, 0.428133923, 0.874992976, 0.52329852,
    //     0.499172047, 0.6028312, 0.095627101, 0.38898065, 0.799446854, 0.618940573,
    //     0.078196824, 0.882892741, 0.844261063, 0.523200747
    // };
    // Read dense0 input from csv file
    std::string dense0Input = "/Users/kateaizpuru/Documents/CNN/test-data/dense0/flatten.csv"; // Replace with your input file path
    float d0Input[D_INPUT];
    loadFunction(dense0Input, 1, D_INPUT, d0Input, true);

    // Output array
    float d0Output[D_OUTPUT];

    //Forward
    forward<D_INPUT, D_OUTPUT>(d0Input, d0Weights, d0Bias, d0Output);
    // DENSE 1
    // File paths
    std::string dense1Weights = "/Users/kateaizpuru/Documents/CNN/test-data/dense1/dense_1_weights.csv"; // Replace with your weights file path
    std::string dense1Bias = "/Users/kateaizpuru/Documents/CNN/test-data/dense1/dense_1_biases.csv";     // Replace with your bias file path
    std::string dense1Input = "/Users/kateaizpuru/Documents/CNN/test-data/dense1/dense1Input.csv"; // Replace with your input file path


    float d1Weights[D1_INPUT * D1_OUTPUT]; //512 (Rows) * 256 (Columns)
    loadFunction(dense1Weights, D1_INPUT, D1_OUTPUT, d1Weights, false);
    std::cout << "Weights loaded" << std::endl;

    float d1Bias[D_BWCOLS]; // 256
    loadFunction(dense1Bias, 1, D1_OUTPUT, d1Bias, true);
    std::cout << "Bias loaded" << std::endl;

    // Read dense1 input from csv file
    float d1Input[D1_INPUT]; 
    loadFunction(dense1Input, D1_INPUT, 1, d1Input, true);
    std::cout << "Input loaded" << std::endl;
    
    // Input array
    float d1Output[D1_OUTPUT];
   
    // Output array
    forward<D1_INPUT, D1_OUTPUT>(d1Input, d1Weights, d1Bias, d1Output);

    // Print output
    printOutput<D_OUTPUT>(d0Output);
    std::cout << "------------------------" << std::endl;
    // printOutput<D1_OUTPUT>(d1Output);

    // Dense1


    return 0;
}
