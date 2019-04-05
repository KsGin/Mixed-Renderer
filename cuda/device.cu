/**
 * File Name : device.cu
 * Author : Yang Fan
 * Date : 2019/4/2
 * define device mixed
 */

#pragma once
#include <device_launch_parameters.h>
#include "../common/device.h"
#include "../common/define.h"
#include "../common/texture.h"
#include <vector>

/*
 * 设置颜色
 */
__device__ void SetPixel(int x, int y, const Color& color, Uint8* pixelColors, int screenWidth, int screenHeight) {
	auto r = color.r;
	auto g = color.g;
	auto b = color.b;
	auto a = color.a;
	CLAMP01(a);
	CLAMP01(b);
	CLAMP01(g);
	CLAMP01(r);

	auto i = (y * screenWidth + x) * 4;

	CLAMP(i , 0 , screenWidth * screenHeight * 4 - 1);

	pixelColors[i - 1] = static_cast<Uint8>(a * 255);
	pixelColors[i - 2] = static_cast<Uint8>(b * 255);
	pixelColors[i - 3] = static_cast<Uint8>(g * 255);
	pixelColors[i - 4] = static_cast<Uint8>(r * 255);
}


/*
 * 渲染管线混合阶段
 */
__global__ void KernelMixed(Pixel* pixels , Color* colors , Uint8* pixelColors , int screenWidth , int screenHeight , int numElements) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < numElements) {

		const int x = pixels[idx].pos._x;
		const int y = pixels[idx].pos._y;	

		SetPixel(x , y , colors[idx] , pixelColors , screenWidth , screenHeight);
	}
}


extern "C" void CallMixed(std::vector<Pixel>& pixels, std::vector<Color>& colors , Uint8* pixelColors , int screenWidth , int screenHeight) {
	const int numPixels = pixels.size();
	const int screenPixelSize = screenWidth * screenHeight * 4;
	
	Pixel* dPixels;
	CUDA_CALL(cudaMalloc(&dPixels , sizeof(Pixel) * numPixels));
	CUDA_CALL(cudaMemset(dPixels , 0 , sizeof(Pixel) * numPixels));
	Color* dColors;
	CUDA_CALL(cudaMalloc(&dColors , sizeof(Color) * numPixels));
	CUDA_CALL(cudaMemset(dColors , 0 , sizeof(Color) * numPixels));
	Uint8* dPixelColors;
	CUDA_CALL(cudaMalloc(&dPixelColors , sizeof(Uint8) * screenPixelSize));
	CUDA_CALL(cudaMemset(dPixelColors , 0 , sizeof(Uint8) * screenPixelSize));

	CUDA_CALL(cudaMemcpy(dPixels , &pixels[0] , sizeof(Pixel) * numPixels , cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dColors , &colors[0] , sizeof(Color) * numPixels , cudaMemcpyHostToDevice));

	// 流水线执行

	KernelMixed<<<(numPixels + 63) / 64 , 64>>>(dPixels , dColors , dPixelColors , screenWidth , screenHeight , numPixels);

	CUDA_CALL(cudaMemcpy(pixelColors , dPixelColors , sizeof(Uint8) * screenPixelSize , cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(dPixels));
	CUDA_CALL(cudaFree(dColors));
	CUDA_CALL(cudaFree(dPixelColors));
}

