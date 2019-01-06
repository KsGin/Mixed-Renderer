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
#include "../common/texture.h"
#include "../cuda/shader.cu"
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
		MODEL , 
		VIEW , 
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
	Shader() {
	}

	/*
	 * Deconstructor
	 */
	~Shader() {
	}

	/*
	 * Set Matrix
	 */
	void setMat(const Math::Matrix& mat , const MatType& type) {
		switch (type) {
			case MODEL: modelMat = mat; break;
			case VIEW: viewMat = mat; break;
			case PERSPECTIVE: perspectiveMat = mat; break;
		}
	}

	/*
	 * Set Texture
	 */
	void setTexture(const Texture& texture , const int idx) {
		this->textures[idx] = texture;
	}

	/*
	 * Vertex Shader
	 */
	__global__ void vertexShader(const VSInput& vsInput , PSInput& psInput){
		auto transMat = modelMat.multiply(viewMat).multiply(perspectiveMat);
		psInput.pos = Math::Matrix::transformCoordinates(vsInput.pos , transMat);
		psInput.normal = Math::Matrix::transform(vsInput.normal, transMat);
		psInput.uv = vsInput.uv;
		psInput.color = vsInput.color;
	}

	/*
	 * Pixel Shader
	 */
	__global__ void pixelShader(PSInput* psInput , Color* color)
	{
		int idx = blockIdx.x * blockDim.x + blockIdx.x;
		color[idx] = textures[0].getPixel(psInput[idx].uv._x , psInput[idx].uv._y);
	}

};