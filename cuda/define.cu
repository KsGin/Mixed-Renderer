/**
 * File Name : define.cu
 * Author : Yang Fan
 * Date : 2018/11/27
 * define some functions
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <assert.h>

/**
 * define a cuda call
 */
#define CUDA_CALL(x) { \
	const cudaError_t e = (x);\
	if(e != cudaSuccess)		\
	{		\
		printf("\nCUDA Error : %s (err_num = %d)\n", cudaGetErrorString(e), e); \
		cudaDeviceReset(); \
		assert(0); \
	}\
}

/**
 * define screen width & height & isFullScreen
 */
#define SCREEN_WIDTH 1920
#define SCREEN_HEIGHT 1080
#define IS_FULL_SCREEN false

/*
 * print device information 
 */
#define PRINT_DEVICE_INFORMATION() { \
    int dc;   \
	cudaGetDeviceCount(&dc); \
	if (dc == 0) { \
		printf("error : no device supporting cuda\n"); \
		exit(1); \
	} \
		\
	int dev = 0;\
	cudaSetDevice(dev);	\
	cudaDeviceProp devProps;	\
	cudaGetDeviceProperties(&devProps, dev); \
	\
	printf("name : %s \ntotalGlobalMem : %zdM\n" , devProps.name , devProps.totalGlobalMem >> 20); \
}


/*
 * defined clamp
 */
#define CLAMP(x , min , max) { \
   if (x > max) x = max;  \
   if (x < min) x = min;  \
}


/*
 * defined clamp 0~1
 */
#define CLAMP01(x) { \
	CLAMP(x , 0 , 1) \
}