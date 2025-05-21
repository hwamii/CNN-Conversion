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

template <typename T,
        size_t IN, size_t OUT,
        ActivationType ACT = RELU>
void dense(const T input[IN], const T weights[IN * OUT], const T bias[OUT], T output[OUT]) {
	T temp[OUT] = {0};

	//Matrix multiplication (both addition and multiplication)
	for (size_t j = 0; j < IN; j++) {
	   for (size_t i = 0; i < OUT; i++) {
#pragma HLS UNROLL factor=8
	        //products[j][i] = input[j] * weights[j * OUT + i];
	        temp[i] += input[j] * weights[j * OUT * i];
	   }
	   // move the ADD_PRODUCT here
	}

	//Add bias here
	for(size_t i = 0; i<OUT; i++){
		temp[i] += bias[i];
	}
    if constexpr (ACT == RELU) {
    	// Richard Code is here
        RL<float, OUT>(temp,output);
    } else if constexpr (ACT == SOFTMAX) {
    	SM<float, OUT>(temp, output);
    } else {
    	transfer_data<T, OUT>(temp, output);
    }
}
