#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <string>
#include <vector>

// Loads a 2D array of weights from a CSV file.
// Expected dimensions: rows x cols.
std::vector<std::vector<float> > loadWeights(const std::string &filename, int rows, int cols);

// Loads biases from a CSV file.
// Expected number of lines equals size.
std::vector<float> loadBias(const std::string &filename, int size);

#endif // WEIGHTS_H