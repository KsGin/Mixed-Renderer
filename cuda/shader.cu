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
#include "../includes/color.hpp"
#include "../includes/math/vector.hpp"
#include <vector>

#define MAX_TEXTURE_SIZE 8 

/****************************************************************通用区************************************************************************************************/
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

/*************************************************************Pixel Shader*************************************************************************************************/
__device__ Color& CubePixelShader(Pixel& pixel , Texture* texture , unsigned char** texturesPixels , Args args) {

	const auto ambient = 0.2;

	const auto directionLight = Math::Vector3(0 , 1 , -1).normalize();
	const auto normal = pixel.normal.normalize();

	auto nd = Math::Vector3::dot(directionLight , normal);

	CLAMP01(nd);

	auto texColor = TexSampler2D(texture[0] , texturesPixels[0] , pixel.uv);
	auto color = texColor * (ambient + nd);

	return color;
}

__device__ Color& WaterPixelShader(Pixel& pixel , Texture* texture , unsigned char** texturesPixels , Args args) {

	const auto ambient = 0.3;

	const auto directionLight = Math::Vector3(0 , 1 , -1).normalize();
	const auto normal = pixel.normal.normalize();

	auto nd = Math::Vector3::dot(directionLight , normal);

	CLAMP01(nd);

	auto uv = pixel.uv / 2 + Math::Vector2(args.bis + 0.25 , 0);

	CLAMP01(uv._x);

	auto texColor = TexSampler2D(texture[0] , texturesPixels[0] , uv);
	auto color = pixel.color * 0.1 + texColor * (ambient + nd);

	return color;
}

/***********************************************************Shader 调用******************************************************************************************************/
__global__ void KernelPixelShader(SHADER_TYPE sType , Color* colors, Pixel* pixels, Texture* textures, unsigned char** texturesPixels, 
                                  const int numElements , Args args) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numElements) {
		switch (sType) {
			case WATER : colors[idx] = WaterPixelShader(pixels[idx] , textures , texturesPixels , args); break;
			case CUBE : colors[idx] = CubePixelShader(pixels[idx] , textures , texturesPixels , args); break;
		}
		
	}
}

extern "C" void CallPixelShader(const std::vector<Pixel>& pixels, const std::vector<Texture>& textures , 
								const SHADER_TYPE& sType , std::vector<Color>& colors, const Args& args) {
	if (pixels.empty()) return;

	const int numPixels = pixels.size();
	const int numTextures = textures.size();

	Pixel* dPixels = nullptr;
	CUDA_CALL(cudaMalloc(&dPixels , sizeof(Pixel) * numPixels));
	CUDA_CALL(cudaMemset(dPixels , 0 , sizeof(Pixel) * numPixels));
	Color* dColors = nullptr;
	CUDA_CALL(cudaMalloc(&dColors , sizeof(Color) * numPixels));
	CUDA_CALL(cudaMemset(dColors , 0 , sizeof(Color) * numPixels));
	Texture* dTextures = nullptr;
	CUDA_CALL(cudaMalloc(&dTextures , sizeof(Texture) * numTextures));
	CUDA_CALL(cudaMemset(dTextures , 0 , sizeof(Texture) * numTextures));

	auto texturePixels = new unsigned char*[MAX_TEXTURE_SIZE];
	auto numAvailableTex = 0;
	for (auto i = 0 ; i < MAX_TEXTURE_SIZE; ++i) {

		const auto tex = textures[i];
		const auto size = tex.width * tex.height * 4;

		if (size == 0) continue;

		unsigned char* texturePixel;

		CUDA_CALL(cudaMalloc(&texturePixel , sizeof(unsigned char) * size));
		CUDA_CALL(cudaMemcpy(texturePixel , tex.pixels, sizeof(unsigned char) * size, cudaMemcpyHostToDevice));

		texturePixels[i] = texturePixel;
		numAvailableTex += 1;
	}

	unsigned char** dTexturePixels = nullptr;
	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&dTexturePixels) , sizeof(unsigned char*) * MAX_TEXTURE_SIZE));
	CUDA_CALL(cudaMemcpy(dTexturePixels , texturePixels , sizeof(unsigned char*) * MAX_TEXTURE_SIZE , cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMemcpy(dPixels , &pixels[0] , numPixels * sizeof(Pixel) , cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dTextures , &textures[0] , numTextures * sizeof(Texture) , cudaMemcpyHostToDevice))

	// 64
	KernelPixelShader<<<(numPixels + 63) / 64 , 64>>>(sType , dColors, dPixels, dTextures, dTexturePixels , numPixels, args);

	CUDA_CALL(cudaMemcpy(&colors[0] , dColors , numPixels * sizeof(Color) , cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(dPixels));
	CUDA_CALL(cudaFree(dColors));
	CUDA_CALL(cudaFree(dTextures));
	CUDA_CALL(cudaFree(dTexturePixels));

	for (auto i = 0 ; i < numAvailableTex; ++i) {
		CUDA_CALL(cudaFree(texturePixels[i]));
	}

	delete[] texturePixels;
}

 