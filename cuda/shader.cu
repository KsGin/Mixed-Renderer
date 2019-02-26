/**
 * File Name : shader.cu
 * Author : Yang Fan
 * Date : 2019/1/6
 * defined shader
 */

#pragma once
#include <device_launch_parameters.h>
#include "../common/define.h"
#include "../common/texture.h"
#include <vector>


__device__ void TexSampler2D(const Texture& texture, const float x, const float y, Color& color)
{
	const int tx = x * texture.width;
	const int ty = y * texture.height;

	auto idx = (ty * texture.width + tx) * 4;

	CLAMP(idx, 4, texture.width * texture.height * 4);

	color.a = texture.pixels[idx - 1] / 255.0f;
	color.b = texture.pixels[idx - 2] / 255.0f;
	color.g = texture.pixels[idx - 3] / 255.0f;
	color.r = texture.pixels[idx - 4] / 255.0f;
}

__global__ void KernelPixelShader(Color* colors , Pixel* pixels  , Texture* textures, const int numElements)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < numElements) 
	{
		colors[idx].r = 1.0f;
		colors[idx].g = 0.0f;
		colors[idx].b = 1.0f;
		colors[idx].a = 1.0f;
		TexSampler2D(textures[0], pixels[idx].uv._x, pixels[idx].uv._y, colors[idx]);
	}
}

extern "C" void CallPixelShader(const std::vector<Pixel>& pixels, const std::vector<Texture>& textures, std::vector<Color>& colors)
{
	const int numPixels = pixels.size();
	const int numTextures = textures.size();

	Pixel* dPixels;
	CUDA_CALL(cudaMalloc(&dPixels , sizeof(Pixel) * numPixels));
	CUDA_CALL(cudaMemset(dPixels , 0 , sizeof(Pixel) * numPixels));
	Color* dColors;
	CUDA_CALL(cudaMalloc(&dColors , sizeof(Color) * numPixels));
	CUDA_CALL(cudaMemset(dColors , 0 , sizeof(Color) * numPixels));
	Texture* dTextures;
	CUDA_CALL(cudaMalloc(&dTextures , sizeof(Texture) * numTextures));
	CUDA_CALL(cudaMemset(dTextures , 0 , sizeof(Texture) * numTextures));

	CUDA_CALL(cudaMemcpy(dPixels , &pixels[0] , numPixels * sizeof(Pixel) , cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dTextures , &textures[0] , numTextures * sizeof(Texture) , cudaMemcpyHostToDevice))

	// 每个线程块执行8个线程
	KernelPixelShader<<<(numPixels + 7) / 8 , 8>>>(dColors, dPixels, dTextures, numPixels);

	CUDA_CALL(cudaMemcpy(&colors[0] , dColors , numPixels * sizeof(Color) , cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(dPixels));
	CUDA_CALL(cudaFree(dColors));
	CUDA_CALL(cudaFree(dTextures));
}

 