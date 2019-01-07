/**
 * File Name : shader.cu
 * Author : Yang Fan
 * Date : 2019/1/6
 * defined shader
 */

#pragma once

#include <device_launch_parameters.h>
#include "../includes/math/matrix.hpp"
#include "../includes/math/vector.hpp"
#include "../common/color.h"
#include "../cuda/define.cu"
#include "../cuda/texture.cu"
#include <vector>

class Shader
{
	/*
	 * modelMat
	 */
	Math::Matrix modelMat = Math::Matrix::identity();
	/*
	 * viewMat
	 */
	Math::Matrix viewMat = Math::Matrix::identity();
	/*
	 * perspectiveMat
	 */
	Math::Matrix perspectiveMat = Math::Matrix::identity();
	/*
	 * textures
	 */
	std::vector<Texture> textures = std::vector<Texture>(8);

public:

	enum MatType
	{
		MODEL,
		VIEW,
		PERSPECTIVE
	};

	struct VSInput
	{
		Math::Vector3 pos;
		Math::Vector3 normal;
		Math::Vector2 uv;
		Color color;
	};

	struct PSInput
	{
		Math::Vector3 pos;
		Math::Vector3 normal;
		Math::Vector2 uv;
		Color color;
	};

	/*
	* Constructor
	*/
	Shader()
	{
	}

	/*
	 * Deconstructor
	 */
	~Shader()
	{
	}

	/*
	 * Set Matrix
	 */
	void setMat(const Math::Matrix& mat, const MatType& type)
	{
		switch (type)
		{
		case MODEL: modelMat = mat;
			break;
		case VIEW: viewMat = mat;
			break;
		case PERSPECTIVE: perspectiveMat = mat;
			break;
		}
	}

	/*
	 * Set Texture
	 */
	void setTexture(const Texture& texture, const int idx)
	{
		this->textures[idx] = texture;
	}

	/*
	 * Vertex Shader
	 */
	void callVertexShader(const VSInput& vsInput, PSInput& psInput)
	{
		auto transMat = modelMat.multiply(viewMat).multiply(perspectiveMat);
		psInput.pos = Math::Matrix::transformCoordinates(vsInput.pos, transMat);
		psInput.normal = Math::Matrix::transform(vsInput.normal, transMat);
		psInput.uv = vsInput.uv;
		psInput.color = vsInput.color;
	}

	/*
	 * Pixel Shader
	 */
	void callPixelShader(std::vector<PSInput> pixels, std::vector<Color> colors);
};


__global__ void pixelShader(Shader::PSInput* psInput, Color* color, Texture* textures)
{
	const int idx = blockIdx.x * blockDim.x + blockIdx.x;
	Sampler2D(textures[0] , psInput[idx].uv._x, psInput[idx].uv._y , color[idx]);
}

void Shader::callPixelShader(std::vector<PSInput> pixels, std::vector<Color> colors)
{
	PSInput* dPixels;
	CUDA_CALL(cudaMalloc((void**)&dPixels , sizeof(Shader::PSInput) * pixels.size()));
	Color* dColors;
	CUDA_CALL(cudaMalloc((void**)&dColors , sizeof(Color) * pixels.size()));
	Texture* dTextures;
	CUDA_CALL(cudaMalloc((void**)&dTextures , sizeof(Texture) * pixels.size()));

	CUDA_CALL(cudaMemcpy(dPixels , &pixels[0] , pixels.size() , cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dTextures , &textures[0] , textures.size() , cudaMemcpyDeviceToDevice))

	pixelShader<<<1 , pixels.size()>>>(dPixels, dColors, dTextures);

	CUDA_CALL(cudaMemcpy(&colors[0] , dColors , pixels.size() , cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaFree(dPixels));
	CUDA_CALL(cudaFree(dColors));
	CUDA_CALL(cudaFree(dTextures));
}
