#include "loadData.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cstdlib>

// This is the implementation file loadData.cpp
// This function aims to load the weights from a CSV file and return a float vector of vectors (2D vector)
// The function takes in the filename, the number of rows and the number of columns 
std::vector<std::vector<float> > loadWeights(const std::string &filename, int csvRows, int csvCols) {
    std::vector<std::vector<float> > weights(csvRows, std::vector<float>(csvCols));
    std::ifstream file(filename); // Read the file
    if (!file.is_open()) { // Check if the file is open
        std::cerr << "Error: Could not open weights file: " << filename << std::endl;
        exit(1);    // Exit the program if the file is not open
    }

    std::string line; // Create a string variable to store the line
    for (int i = 0; i < csvRows; i++) { // Loop through the rows
        if (!std::getline(file, line)) { // Get the line from the file
            std::cerr << "Error: Not enough lines in weights file." << std::endl;
            exit(1);
        }
        std::stringstream ss(line); // Create a stringstream object to parse the line
        std::string value; // Create a string variable to store the value
        int j = 0;
        while (std::getline(ss, value, ',')) { // Loop through the values in the line with a comma delimiter
            try {
                weights[i][j] = std::stof(value);
                std::cout << "Weights[" << i << "][" << j << "] = " << weights[i][j] << std::endl; // Debug print
            } catch (const std::invalid_argument &e) {
                std::cerr << "Error: Invalid number at row " << i << ", column " << j << std::endl;
                exit(1);
            }
            j++;
            if (j > csvCols) {
                std::cerr << "Error: Too many values in row " << i << " of weights file." << std::endl;
                exit(1);
            }
        }
    }
    file.close(); // This was read successfully, so close the file
    return weights;
}

std::vector<float> loadBias(const std::string &filename, int size) {
    std::vector<float> bias(size);
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open bias file: " << filename << std::endl;
        exit(1);
    }

    std::string line;
    for (int i = 0; i < size; i++) {
        if (!std::getline(file, line)) {
            std::cerr << "Error: Not enough lines in bias file." << std::endl;
            exit(1);
        }
        
        std::stringstream ss(line);
        ss >> bias[i];
        // std::cout << "Bias[" << i << "] = " << bias[i] << std::endl; // I know that the bias file was read correctly
    }

    file.close();
    return bias;
}

// int main() {
//     std::string filename = "/Users/kateaizpuru/Documents/CNN/test-data/weights.csv";
//     std::string biasFile = "/Users/kateaizpuru/Documents/CNN/test-data/biases.csv";
//     int rows = 16;
//     int cols = 512;

//     std::vector<std::vector<float> > weights = loadWeights(filename, rows, cols);
//     std::vector<float> biases = loadBias(biasFile, cols);

//     return 0;
// }