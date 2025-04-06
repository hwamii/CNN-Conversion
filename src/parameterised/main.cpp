#include "loadFunction.h"
#include "denseLayer.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstddef>
// Define constants for array sizes
constexpr std::size_t W_EXP_RNUM = 16;  // Number of rows in weights matrix (input size)
constexpr std::size_t W_B_CNUM = 512;   // Number of columns in weights matrix (output size)
constexpr std::size_t B_ROWS = 512;     // Number of elements in bias array
constexpr std::size_t B_COL = 1;       // Number of rows in bias array
// constexpr std::size_t W_ROWS = 16;      // Number of rows in weights matrix

int main() {
    
    // File paths
    std::string weightsFile = "/Users/kateaizpuru/Documents/CNN/test-data/weights.csv"; // Replace with your weights file path
    std::string biasFile = "/Users/kateaizpuru/Documents/CNN/test-data/biases.csv"; // Replace with your bias file path

    try {
        // Load weights and biases from CSV files

        float weights[W_EXP_RNUM * W_B_CNUM];
        loadFunction(weightsFile, W_EXP_RNUM, W_B_CNUM, weights,false);
        

        // float* weights = loadFunction(weightsFile, W_EXP_RNUM, W_B_CNUM, false);


        float bias[B_ROWS];
        loadFunction(biasFile, B_ROWS, B_COL, bias, true);
    
        // float* bias = loadFunction(biasFile, B_ROWS, B_COL, true);
        
        // Input array
        float d_in[W_EXP_RNUM] = {
            0.023172325, 0.954666768, 0.537868863, 0.428133923, 0.874992976, 0.52329852,
        0.499172047, 0.6028312, 0.095627101, 0.38898065, 0.799446854, 0.618940573,
        0.078196824, 0.882892741, 0.844261063, 0.523200747
    };


        // Output array
        float d_out[W_B_CNUM];

        // Perform forward pass
        forward<16, 512>(d_in, weights, bias, d_out);
        // forward<W_EXP_RNUM, W_B_CNUM>(d_in, weights, bias, d_out);

        // Print output
        printOutput<W_B_CNUM>(d_out);

        // Free dynamically allocated memory
       
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}