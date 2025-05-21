#pragma once

#include <iostream>
#include <iomanip>
#include <cstddef> // For size_t
#include <cmath>
#include <cstring>
enum ActivationType { NONE, RELU, SOFTMAX };
template <typename T, size_t DIM>
void transfer_data(T inData[DIM], T outData[DIM]){
	memcpy(outData,inData,sizeof(T)*DIM);
}

template <typename T,size_t DIM>
void RL(T* inData,T* outData){
	for (size_t i = 0; i < DIM; i++) {
		outData[i] = (inData[i] > (T)0) ? inData[i] : (T)0;
	}
}

template<typename T, size_t DIM>
void SM(T* inData, T* outData){
	T maxVal = inData[0];
	for (size_t i = 1; i < DIM; i++) {
	    if (inData[i] > maxVal) maxVal = inData[i];
	}

	// Compute exponentials and sum
	T sumExp = (T)0;
	for (size_t i = 0; i < DIM; i++) {
	    outData[i] = (sizeof(T) == 4
	        ? expf(inData[i] - maxVal)
	        : exp(inData[i] - maxVal));
	    sumExp += outData[i];
	}

	// Normalize
	for (size_t i = 0; i < DIM; i++) {
	    outData[i] /= sumExp;
	}
}

template <typename T, size_t IN, size_t OUT, ActivationType ACT = RELU>
void dense(const T input[IN], const T weights[IN * OUT], const T bias[OUT], T output[OUT]) {
    T temp[OUT] = {0};  // Initialize to zero

    for (size_t j = 0; j < IN; j++) {
    #pragma HLS UNROLL factor=8
        for (size_t i = 0; i < OUT; i++) {
        #pragma HLS PIPELINE
            temp[i] += input[j] * weights[j * OUT + i];
        }
    }

    // Add bias & apply activation
    if constexpr (ACT == RELU) {
        for (size_t i = 0; i < OUT; i++) {
            output[i] = (temp[i] + bias[i] > 0) ? temp[i] + bias[i] : 0;
        }
    } else if constexpr (ACT == SOFTMAX) {
        SM<T, OUT>(temp, output);  // Softmax includes bias internally
    } else {
        for (size_t i = 0; i < OUT; i++) {
            output[i] = temp[i] + bias[i];
        }
    }

}

