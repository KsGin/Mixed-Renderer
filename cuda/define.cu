/**
 * File Name : define.cuh
 * Author : Yang Fan
 * Date : 2018/11/27
 * define some functions with cuda
 */

#pragma once

/**
 * define a cuda call
 */
#define CUDA_CALL(x) { \
	const cudaError_t e = (x);	\
	if (e != cudaSuccess) \							\
	{							\
		printf("\nCUDA Error : %s (err_num = %d)\n" , cudaGetErrorString(e) , e);					\
		cudaDeviceReset();						\
		assert(0);							\
	}                  \
}
