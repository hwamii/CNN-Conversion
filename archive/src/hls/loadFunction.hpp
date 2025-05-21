#pragma once

#include <iostream>
#include <fstream>
#include <sstream>

void loadFunction(const std::string &filename, int csvRows, int csvCols, float * data, bool isBias) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        exit(1);
    }

    std::string line;
    for (int i = 0; i < csvRows; i++) {
        if (!std::getline(file, line)) {
            std::cerr << "Error: Not enough lines in file." << std::endl;
            exit(1);
        }

        std::stringstream ss(line);
        std::string value;
        int j = 0;

        while (std::getline(ss, value, ',')) {
            try {
                if (isBias) {
                    data[j] = std::stof(value);  // bias: flat 1D array
                } else {
                    data[i * csvCols + j] = std::stof(value);  // weights or input matrix
                }
            } catch (const std::invalid_argument &e) {
                std::cerr << "Error: Invalid number at row " << i << ", column " << j << std::endl;
                exit(1);
            }
            j++;
        }

        if (j < csvCols) {
            std::cerr << "Error: Too few values in row " << i << " of file." << std::endl;
            exit(1);
        }
    }

    file.close();
}

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

// Define freeData function
void freeData(float* data) {
    delete[] data; // Delete the 1D array
}