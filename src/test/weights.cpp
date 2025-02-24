#include "weights.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>

std::vector<std::vector<float>> loadWeights(const std::string &filename, int rows, int cols) {
    std::vector<std::vector<float>> weights(rows, std::vector<float>(cols, 0.0f));
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open weights file: " << filename << std::endl;
        exit(1);
    }
    
    std::string line;
    for (int i = 0; i < rows; i++) {
        if (!std::getline(file, line)) {
            std::cerr << "Error: Not enough lines in weights file." << std::endl;
            exit(1);
        }
        
        std::cout << "Reading line " << i << ": " << line << std::endl;
        
        std::stringstream ss(line);
        std::string value;
        int j = 0;
        while (std::getline(ss, value, ',')) {
            if (j >= cols) {
                std::cerr << "Error: Too many values in row " << i << " of weights file." << std::endl;
                exit(1);
            }
            try {
                weights[i][j] = std::stof(value);
            } catch (const std::invalid_argument &e) {
                std::cerr << "Error: Invalid number in weights file at row " << i 
                          << ", column " << j << " (value: '" << value << "')" << std::endl;
                exit(1);
            }
            j++;
        }
        if (j < cols) {
            std::cerr << "Error: Too few values in row " << i << " of weights file." << std::endl;
            exit(1);
        }
    }
    
    file.close();
    return weights;
}

std::vector<float> loadBias(const std::string &filename, int size) {
    std::vector<float> biases(size, 0.0f);
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open bias file: " << filename << std::endl;
        exit(1);
    }
    
    std::string line;
    int i = 0;
    while (std::getline(file, line) && i < size) {
        try {
            biases[i] = std::stof(line);
        } catch (const std::invalid_argument &e) {
            std::cerr << "Error: Invalid number in bias file at line " << i 
                      << " (value: '" << line << "')" << std::endl;
            exit(1);
        }
        i++;
    }
    if (i < size) {
        std::cerr << "Error: Not enough lines in bias file." << std::endl;
        exit(1);
    }
    
    file.close();
    return biases;
}