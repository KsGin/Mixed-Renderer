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


__device__ Color& TexSampler2D(const Texture& texture, unsigned char* texturesPixels , const Math::Vector2& uv) {

	Color color;

	const int tx = uv._x * texture.width;
	const int ty = uv._y * texture.height;

	auto idx = (ty * texture.width + tx) * 4;

	CLAMP(idx, 4, texture.width * texture.height * 4);

	color.a = texturesPixels[idx - 1] / 255.0f;
	color.b = texturesPixels[idx - 2] / 255.0f;
	color.g = texturesPixels[idx - 3] / 255.0f;
	color.r = texturesPixels[idx - 4] / 255.0f;

	return color;
}

/*
 * Pixel Shader
 */
__device__ Color& PixelShader(Pixel& pixel , const Texture& texture , unsigned char* texturesPixels) {
	// 环境光
	const auto ambient = 0.1;

	auto texColor = TexSampler2D(texture, texturesPixels, pixel.uv);

	const auto directionLight = Math::Vector3(0 , 1 , -1).normalize();
	const auto normal = pixel.normal.normalize();

	auto nd = Math::Vector3::dot(directionLight , normal);

	CLAMP01(nd);

	auto color = texColor * (ambient + nd);

	return color;
}

__global__ void KernelPixelShader(Color* colors, Pixel* pixels, Texture* textures, unsigned char* texturesPixels,
                                  const int numElements) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numElements) {
		colors[idx] = PixelShader(pixels[idx] , textures[0] , texturesPixels);
	}
}

extern "C" void CallPixelShader(const std::vector<Pixel>& pixels, const std::vector<Texture>& textures,
                                std::vector<Color>& colors) {
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

	// 以下拷贝第一个纹理的 pixels 数组
	unsigned char* dTexturesPixels;
	CUDA_CALL(cudaMalloc(&dTexturesPixels, sizeof(unsigned char) * textures[0].width * textures[0].height * 4));
	CUDA_CALL(cudaMemcpy(dTexturesPixels, textures[0].pixels, sizeof(unsigned char) * textures[0].width * textures[0].height * 4, cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMemcpy(dPixels , &pixels[0] , numPixels * sizeof(Pixel) , cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dTextures , &textures[0] , numTextures * sizeof(Texture) , cudaMemcpyHostToDevice))

	// 64
	KernelPixelShader<<<(numPixels + 15) / 16 , 16>>>(dColors, dPixels, dTextures, dTexturesPixels, numPixels);

	CUDA_CALL(cudaMemcpy(&colors[0] , dColors , numPixels * sizeof(Color) , cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(dPixels));
	CUDA_CALL(cudaFree(dColors));
	CUDA_CALL(cudaFree(dTextures));
	CUDA_CALL(cudaFree(dTexturesPixels));
}

 