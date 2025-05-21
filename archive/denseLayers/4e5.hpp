#pragma once

#include <iostream>
#include <iomanip>
#include <cstddef> // For size_t
#include <cmath>
#include <string>
enum ActivationType { NONE, RELU, SOFTMAX };
template <typename T, size_t DIM>
void transfer_data(T inData[DIM], T outData[DIM]){
	memcpy(outData,inData,sizeof(T)*DIM);
}

template <typename T,size_t DIM>
void RELU(T* inData,T* outData){
	for (size_t i = 0; i < DIM; i++) {
		outData[i] = (inData[i] > (T)0) ? inData[i] : (T)0;
	}

}

template<typename T, size_t DIM>
void SOFTMAX(T* inData, T* outData){
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
	// TODO: Remove products[IN]
	// Richard wants this: T products[OUT];
	T products[IN][OUT]; // Richard does not like this.
	T temp[OUT];
	//Compute input[j] * weights[j][i] and store
	for (size_t j = 0; j < IN; j++) {
	   for (size_t i = 0; i < OUT; i++) {
#pragma HLS UNROLL factor=8
	        products[j][i] = input[j] * weights[j * OUT + i];
	   }
	   // move the ADD_PRODUCT here

	}
	// Sum across j for each output neuron
	for (size_t i = 0; i < OUT; i++) {
#pragma HLS PIPELINE
	   T sum = 0;
	   ADD_PRODUCT:for (size_t j = 0; j < IN; j++) {
#pragma HLS UNROLL factor=8
	     sum += products[j][i];
	   }
	   temp[i] = sum + bias[i];


	   //Putting the activation layers here anda treat it as a element.
//	   RELU<float, 1>(&temp[i],&output[i]);
	}
    if constexpr (ACT == RELU) {
    	// Richard Code is here

        RELU<float, OUT>(temp,output);
    } else if constexpr (ACT == SOFTMAX) {
    	SOFTMAX<float, OUT>(temp, output);
    } else {
        for (size_t i = 0; i < OUT; i++) {
            output[i] = temp[i];
        }
    }
}
