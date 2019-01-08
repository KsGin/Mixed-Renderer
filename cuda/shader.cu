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

	CLAMP(idx, 0, texture.width * texture.height * 4 - 1);

	color.a = texture.pixels[idx - 1] / 255.0f;
	color.b = texture.pixels[idx - 2] / 255.0f;
	color.g = texture.pixels[idx - 3] / 255.0f;
	color.r = texture.pixels[idx - 4] / 255.0f;
}

__global__ void GlobalPixelShader(PSInput* psInput, Texture* textures ,Color* color)
{
	const int idx = blockIdx.x * blockDim.x + blockIdx.x;
	TexSampler2D(textures[0], psInput[idx].uv._x, psInput[idx].uv._y, color[idx]);
}

extern "C" void CallGlobalPixelShader(const std::vector<PSInput> pixels, const std::vector<Texture> textures, std::vector<Color> colors)
{
	PSInput* dPixels;
	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&dPixels) , sizeof(PSInput) * pixels.size()));
	CUDA_CALL(cudaMemset(dPixels , 0 , sizeof(PSInput) * pixels.size()));
	Color* dColors;
	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&dColors) , sizeof(Color) * pixels.size()));
	CUDA_CALL(cudaMemset(dColors , 0 , sizeof(Color) * pixels.size()));
	Texture* dTextures;
	CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&dTextures) , sizeof(Texture) * textures.size()));
	CUDA_CALL(cudaMemset(dTextures , 0 , sizeof(Texture) * textures.size()));

	CUDA_CALL(cudaMemcpy(dPixels , &pixels[0] , pixels.size() * sizeof(PSInput) , cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dTextures , &textures[0] , textures.size() * sizeof(Texture) , cudaMemcpyHostToDevice))

	GlobalPixelShader<<<1 , pixels.size()>>>(dPixels, dTextures, dColors);

	if (colors.size() < pixels.size())
	{
		colors.resize(pixels.size() + 1);
	}

	CUDA_CALL(cudaMemcpy(&colors[0] , dColors , colors.size() * sizeof(Color) , cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(dPixels));
	CUDA_CALL(cudaFree(dColors));
	CUDA_CALL(cudaFree(dTextures));
}

