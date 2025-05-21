#pragma once
#include <string>

struct LayerPaths {
    std::string input;
    std::string weights;
    std::string bias;
    std::string output;
};

struct ModelPaths {
    LayerPaths dense0 = {
        "/Users/kateaizpuru/Documents/CNN/src/testData/flatten.csv",
        "/Users/kateaizpuru/Documents/CNN/src/testData/dense/dense_weights.csv",
        "/Users/kateaizpuru/Documents/CNN/src/testData/dense/dense_biases.csv",
        "/Users/kateaizpuru/Documents/CNN/src/testData/dense/dense_output.csv"      
    };

    LayerPaths dense1 = {
        "/Users/kateaizpuru/Documents/CNN/src/testData/dense/dense_output.csv",
        "/Users/kateaizpuru/Documents/CNN/src/testData/dense1/dense_1_weights.csv",
        "/Users/kateaizpuru/Documents/CNN/src/testData/dense1/dense_1_biases.csv",
        "/Users/kateaizpuru/Documents/CNN/src/testData/dense1/dense1_output.csv"
    };

    LayerPaths dense2 = {
        "/Users/kateaizpuru/Documents/CNN/src/testData/dense1/dense1_output.csv",
        "/Users/kateaizpuru/Documents/CNN/src/testData/dense2/dense_2_weights.csv",
        "/Users/kateaizpuru/Documents/CNN/src/testData/dense2/dense_2_biases.csv",
        "/Users/kateaizpuru/Documents/CNN/src/testData/dense2/dense2_output.csv"
    };

    LayerPaths dense3 = {
        "/Users/kateaizpuru/Documents/CNN/src/testData/dense2/dense2_output.csv",
        "/Users/kateaizpuru/Documents/CNN/src/testData/dense3/dense_3_weights.csv",
        "/Users/kateaizpuru/Documents/CNN/src/testData/dense3/dense_3_biases.csv",
        "/Users/kateaizpuru/Documents/CNN/src/testData/dense3/dense3_output.csv"
    };
};