#ifndef LOADDATA_H
#define LOADDATA_H

#include <vector>
#include <string>

// Function declarations
std::vector<std::vector<float> > loadWeights(const std::string &filename, int csvRows, int csvCols);
std::vector<float> loadBias(const std::string &filename, int size);

#endif