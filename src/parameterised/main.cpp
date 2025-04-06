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
        float weights2D[W_EXP_RNUM][W_B_CNUM];
        loadFunction(weightsFile, W_EXP_RNUM, W_B_CNUM, weights,false);
        
        float bias[B_ROWS];
        loadFunction(biasFile, B_ROWS, B_COL, bias, true);

        // Input array
        float d_in[W_EXP_RNUM] = {
            0.7905826, 0.95574766, 0.17418092, 0.23858057, 0.03401104, 0.03165761, 0.14767082, 0.46765873, 0.06522544, 0.0605169, 
            0.57054, 0.30463323, 0.5606492, 0.4245618, 0.17499015, 0.6956044
        };

        // Output array
        float d_out[W_B_CNUM];

        // Perform forward pass
        forward<W_EXP_RNUM, W_B_CNUM>(d_in, weights, bias, d_out);

        // Print output
        printOutput<W_B_CNUM>(d_out);

        // Free dynamically allocated memory
       
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}