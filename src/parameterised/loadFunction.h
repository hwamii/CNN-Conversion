#pragma once

#include <string>

void loadFunction(const std::string &filename, int csvRows, int csvCols, float * data, bool isBias);
// float* loadFunction(const std::string &filename, int csvRows, int csvCols, bool isBias);

void freeData(float* data);