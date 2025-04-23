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
        "/home/jaizpuru/ML-Conversion/hls_template/testData/flatten.csv",
        "/home/jaizpuru/ML-Conversion/hls_template/testData/dense/dense_weights.csv",
        "/home/jaizpuru/ML-Conversion/hls_template/testData/dense/dense_biases.csv",
        "/home/jaizpuru/ML-Conversion/hls_template/testData/dense/dense_output.csv"
    };

    LayerPaths dense1 = {
        "/home/jaizpuru/ML-Conversion/hls_template/testData/dense/dense_output.csv",
        "/home/jaizpuru/ML-Conversion/hls_template/testData/dense1/dense_1_weights.csv",
        "/home/jaizpuru/ML-Conversion/hls_template/testData/dense1/dense_1_biases.csv",
        "/home/jaizpuru/ML-Conversion/hls_template/testData/dense1/dense1_output.csv"
    };

    LayerPaths dense2 = {
        "/home/jaizpuru/ML-Conversion/hls_template/testData/dense1/dense1_output.csv",
        "/home/jaizpuru/ML-Conversion/hls_template/testData/dense2/dense_2_weights.csv",
        "/home/jaizpuru/ML-Conversion/hls_template/testData/dense2/dense_2_biases.csv",
        "/home/jaizpuru/ML-Conversion/hls_template/testData/dense2/dense2_output.csv"
    };

    LayerPaths dense3 = {
        "/home/jaizpuru/ML-Conversion/hls_template/testData/dense2/dense2_output.csv",
        "/home/jaizpuru/ML-Conversion/hls_template/testData/dense3/dense_3_weights.csv",
        "/home/jaizpuru/ML-Conversion/hls_template/testData/dense3/dense_3_biases.csv",
        "/home/jaizpuru/ML-Conversion/hls_template/testData/dense3/dense3_output.csv"
    };
};