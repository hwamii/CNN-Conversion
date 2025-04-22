
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


// Function to load data into a dynamically allocated C-style array
float** loadData(const std::string &filename, int csvRows, int csvCols, bool isBias = false) {
    // Dynamically allocate memory for the 2D array
    float** data = new float*[csvRows];
    for (int i = 0; i < csvRows; i++) {
        data[i] = new float[csvCols];
    }

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
                ss >> data[i][0]; // Directly convert the line to a float
                std::cout << "Data[" << i << "][0] = " << data[i][0] << std::endl; // Debug print
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
                    data[i][j] = std::stof(value);
                    std::cout << "Data[" << i << "][" << j << "] = " << data[i][j] << std::endl; // Debug print
                } catch (const std::invalid_argument &e) {
                    std::cerr << "Error: Invalid number at row " << i << ", column " << j << std::endl;
                    exit(1);
                }
                j++;
                if (j >= csvCols) { // Stop reading more values if we've reached the expected number of columns
                    break;
                }
            }
            if (j < csvCols) { // Check if there are fewer values than expected
                std::cerr << "Error: Too few values in row " << i << " of file." << std::endl;
                exit(1);
            }
        }
    }
    file.close(); // This was read successfully, so close the file
    return data;
}

// Function to free the dynamically allocated memory
void freeData(float** data, int csvRows) {
    for (int i = 0; i < csvRows; i++) {
        delete[] data[i]; // Delete
    }
    delete[] data; // Delete
}

// Example main function to test the loadData function
// int main() {
//     // Example usage for weights file [16, 512]
//     int weightsRows = 16;
//     int weightsCols = 512;
//     std::string weightsFile = "/Users/kateaizpuru/Documents/CNN/test-data/weights.csv";// Replace with your weights file name
//     float** weightsData = loadData(weightsFile, weightsRows, weightsCols);

//     // Example usage for bias file [512, 1]
//     int biasRows = 512;
//     int biasCols = 1;
//     std::string biasFile = "/Users/kateaizpuru/Documents/CNN/test-data/biases.csv"; // Replace with your bias file name
//     float** biasData = loadData(biasFile, biasRows, biasCols, true);

//     // Print the loaded data (optional)
//     std::cout << "Weights data:" << std::endl;
//     for (int i = 0; i < weightsRows; i++) {
//         for (int j = 0; j < weightsCols; j++) {
//             std::cout << "Weights[" << i << "][" << j << "] = " << weightsData[i][j] << std::endl;
//         }
//     }

//     std::cout << "Bias data:" << std::endl;
//     for (int i = 0; i < biasRows; i++) {
//         std::cout << "Bias[" << i << "][0] = " << biasData[i][0] << std::endl;
//     }

//     // Free the dynamically allocated memory
//     freeData(weightsData, weightsRows);
//     freeData(biasData, biasRows);

//     return 0;
// }
