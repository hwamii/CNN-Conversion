#ifndef LOADFUNCTION_H
#define LOADFUNCTION_H


#include <string>

// Function declarations
float* loadFunction(const std::string &filename, int csvRows, int csvCols, bool isBias);
void freeData(float* data);

#endif