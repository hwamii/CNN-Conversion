#pragma once

#include <string>

void loadFunction(const std::string &filename, int csvRows, int csvCols, float * data, bool isBias);
// float* loadFunction(const std::string &filename, int csvRows, int csvCols, bool isBias);
void saveToCSV(const std::string &filename, const float *data, std::size_t length);

void freeData(float* data);