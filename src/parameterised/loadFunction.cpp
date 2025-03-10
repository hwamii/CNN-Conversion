#include "loadFunction.h"
#include <iostream>
#include <fstream>
#include <sstream>

// Define loadData function
float* loadFunction(const std::string &filename, int csvRows, int csvCols, bool isBias) {
    // Dynamically allocate memory for the 1D array
    float* data = new float[csvRows * csvCols];

    std::ifstream file(filename); // Read the file
    if (!file.is_open()) { // Check if the file is open
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        exit(1); // Exit the program if the file is not open
    }

    std::string line; // Create a string variable to store the line
    for (int i = 0; i < csvRows; i++) { // Loop through the rows
        if (!std::getline(file, line)) { // Get the line from the file
            std::cerr << "Error: Not enough lines in file." << std::endl;
            exit(1);
        }

        if (isBias) {
            // For bias files, each line contains a single value
            try {
                std::stringstream ss(line);
                ss >> data[i]; // Directly convert the line to a float
                std::cout << "Data[" << i << "][0] = " << data[i] << std::endl; // Debug print
            } catch (const std::invalid_argument &e) {
                std::cerr << "Error: Invalid number at row " << i << std::endl;
                exit(1);
            }
        } else {
            // For weights files, each line contains multiple comma-separated values
            std::stringstream ss(line); // Create a stringstream object to parse the line
            std::string value; // Create a string variable to store the value
            int j = 0;
            while (std::getline(ss, value, ',')) { // Loop through the values in the line with a comma delimiter
                try {
                    data[i * csvCols + j] = std::stof(value); // Flatten the 2D array into 1D
                    std::cout << "Data[" << i << "][" << j << "] = " << data[i * csvCols + j] << std::endl; // Debug print
                } catch (const std::invalid_argument &e) {
                    std::cerr << "Error: Invalid number at row " << i << ", column " << j << std::endl;
                    exit(1);
                }
                j++;
                if (j >= csvCols) { // Stop reading more values if we've reached the csvCols
                    break;
                }
            }
            if (j < csvCols) { // Check if there are fewer values than expected
                std::cerr << "Error: Too few values in row " << i << " of file." << std::endl;
                exit(1);
            }
        }
    }
    file.close(); 
    return data;
}

// Define freeData function
void freeData(float* data) {
    delete[] data; // Delete the 1D array
}