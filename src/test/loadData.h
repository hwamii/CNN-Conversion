#ifndef LOADDATA_H
#define LOADDATA_H

#include <string>

// Function declarations
float** loadData(const std::string &filename, int csvRows, int csvCols, bool isBias);
void freeData(float** data, int csvRows);

#endif