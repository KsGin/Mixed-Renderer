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
#include <unordered_map>

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

	// const auto ambient = 0.3;
	//
	// const auto directionLight = Math::Vector3(0 , 1 , -1).normalize();
	// const auto normal = pixel.normal.normalize();
	//
	// auto nd = Math::Vector3::dot(directionLight , normal);
	//
	// CLAMP01(nd);

	auto texColor = TexSampler2D(texture[0] , texturesPixels[0] , pixel.uv);
	// auto color = texColor * ambient;

	return texColor;
}

__device__ Color& WaterPixelShader(Pixel& pixel , Texture* texture , unsigned char** texturesPixels , Args args) {

	// const auto ambient = 0.3;
	//
	// const auto directionLight = Math::Vector3(0 , 1 , -1).normalize();
	// const auto normal = pixel.normal.normalize();
	//
	// auto nd = Math::Vector3::dot(directionLight , normal);
	//
	// CLAMP01(nd);

	auto uv = pixel.uv / 2 + Math::Vector2(args.bis + 0.25 , 0);

	CLAMP01(uv._x);

	auto texColor = TexSampler2D(texture[0] , texturesPixels[0] , uv);
	// auto color = pixel.color * 0.1 + texColor * ambient;

	return texColor;
}


/***********************************************************Shader 调用******************************************************************************************************/
__global__ void KernelPixelShader(Color* colors, Pixel* pixels, Texture* cubeTextures, unsigned char** cubeTexturesPixels, 
                                  Args cubeArgs , Texture* waterTextures, unsigned char** waterTexturesPixels, Args waterArgs , const int numElements ) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numElements) {
		switch (pixels[idx].sType) {
			case CUBE : colors[idx] = CubePixelShader(pixels[idx] , cubeTextures , cubeTexturesPixels , cubeArgs); break;
			case WATER : colors[idx] = WaterPixelShader(pixels[idx] , waterTextures , waterTexturesPixels , waterArgs); break;
			case LIGHT : colors[idx] = Color::white(); break;
		}
		
	}
}

extern "C" void CallPixelShader(std::unordered_map<ShaderType , ShaderData>& sDataMaps , const std::vector<Pixel>& pixels, std::vector<Color>& colors) {
	if (pixels.empty()) return;

	const int numPixels = pixels.size();

	Pixel* dPixels = nullptr;
	CUDA_CALL(cudaMalloc(&dPixels , sizeof(Pixel) * numPixels));
	CUDA_CALL(cudaMemset(dPixels , 0 , sizeof(Pixel) * numPixels));
	Color* dColors = nullptr;
	CUDA_CALL(cudaMalloc(&dColors , sizeof(Color) * numPixels));
	CUDA_CALL(cudaMemset(dColors , 0 , sizeof(Color) * numPixels));

	CUDA_CALL(cudaMemcpy(dPixels , &pixels[0] , numPixels * sizeof(Pixel) , cudaMemcpyHostToDevice));

	// CUBE SHADER
	Texture* dCubeTextures = nullptr;
	CUDA_CALL(cudaMalloc(&dCubeTextures , sizeof(Texture) * MAX_TEXTURE_SIZE));
	CUDA_CALL(cudaMemset(dCubeTextures , 0 , sizeof(Texture) * MAX_TEXTURE_SIZE));

	auto numCubeAvailableTex = 0;
	auto cubeTexturePixels = new unsigned char*[MAX_TEXTURE_SIZE];
	for (auto i = 0 ; i < MAX_TEXTURE_SIZE; ++i) {
	
		const auto tex = sDataMaps[CUBE].textures[i];
		const auto size = tex.width * tex.height * 4;
	
		if (size == 0) continue;
	
		unsigned char* texturePixel;
	
		CUDA_CALL(cudaMalloc(&texturePixel , sizeof(unsigned char) * size));
		CUDA_CALL(cudaMemcpy(texturePixel , tex.pixels, sizeof(unsigned char) * size, cudaMemcpyHostToDevice));
	
		cubeTexturePixels[i] = texturePixel;
		++numCubeAvailableTex;
	}
	
	unsigned char** dCubeTexturePixels = nullptr;
	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&dCubeTexturePixels) , sizeof(unsigned char*) * MAX_TEXTURE_SIZE));
	CUDA_CALL(cudaMemcpy(dCubeTexturePixels , cubeTexturePixels , sizeof(unsigned char*) * MAX_TEXTURE_SIZE , cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dCubeTextures , &sDataMaps[CUBE].textures[0] , MAX_TEXTURE_SIZE * sizeof(Texture) , cudaMemcpyHostToDevice))

	//WATER SHADER
	Texture* dWaterTextures = nullptr;
	CUDA_CALL(cudaMalloc(&dWaterTextures , sizeof(Texture) * MAX_TEXTURE_SIZE));
	CUDA_CALL(cudaMemset(dWaterTextures , 0 , sizeof(Texture) * MAX_TEXTURE_SIZE));

	auto numWaterAvailableTex = 0;
	auto waterTexturePixels = new unsigned char*[MAX_TEXTURE_SIZE];
	for (auto i = 0 ; i < MAX_TEXTURE_SIZE; ++i) {
	
		const auto tex = sDataMaps[WATER].textures[i];
		const auto size = tex.width * tex.height * 4;
	
		if (size == 0) continue;
	
		unsigned char* texturePixel;
	
		CUDA_CALL(cudaMalloc(&texturePixel , sizeof(unsigned char) * size));
		CUDA_CALL(cudaMemcpy(texturePixel , tex.pixels, sizeof(unsigned char) * size, cudaMemcpyHostToDevice));
	
		waterTexturePixels[i] = texturePixel;
		++numWaterAvailableTex;
	}
	
	unsigned char** dWaterTexturePixels = nullptr;
	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&dWaterTexturePixels) , sizeof(unsigned char*) * MAX_TEXTURE_SIZE));
	CUDA_CALL(cudaMemcpy(dWaterTexturePixels , waterTexturePixels , sizeof(unsigned char*) * MAX_TEXTURE_SIZE , cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dWaterTextures , &sDataMaps[WATER].textures[0] , MAX_TEXTURE_SIZE * sizeof(Texture) , cudaMemcpyHostToDevice))
	
	
	// 64
	KernelPixelShader<<<(numPixels + 63) / 64 , 64>>>(dColors, dPixels, dCubeTextures, dCubeTexturePixels, sDataMaps[CUBE].args , dWaterTextures, dWaterTexturePixels , sDataMaps[WATER].args , numPixels);
	
	cudaDeviceSynchronize();
	
	CUDA_CALL(cudaMemcpy(&colors[0] , dColors , numPixels * sizeof(Color) , cudaMemcpyDeviceToHost));
	
	CUDA_CALL(cudaFree(dPixels));
	CUDA_CALL(cudaFree(dColors));
	CUDA_CALL(cudaFree(dCubeTextures));
	CUDA_CALL(cudaFree(dCubeTexturePixels));
	CUDA_CALL(cudaFree(dWaterTextures));
	CUDA_CALL(cudaFree(dWaterTexturePixels));
	
	for (auto i = 0 ; i < numCubeAvailableTex; ++i) {
		CUDA_CALL(cudaFree(cubeTexturePixels[i]));
	}

	for (auto i = 0 ; i < numWaterAvailableTex; ++i) {
		CUDA_CALL(cudaFree(waterTexturePixels[i]));
	}
	
	delete[] cubeTexturePixels;
	delete[] waterTexturePixels;
}

 