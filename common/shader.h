/**
 * File Name : shader.h
 * Author : Yang Fan
 * Date : 2019/1/7
 * defined shader
 */

#pragma once

#include "../includes/math/matrix.hpp"
#include "../common/define.h"
#include "../common/texture.h"
#include <vector>

extern "C" void CallPixelShader(const std::vector<Pixel>& pixels,const std::vector<Texture>& textures, std::vector<Color>& colors);

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
	void vertexShader(const Vertex& vsInput, Pixel& psInput)
	{
		auto transMat = modelMat.multiply(viewMat).multiply(perspectiveMat);
		psInput.pos = Math::Matrix::transformCoordinates(vsInput.pos, transMat);
		psInput.normal = Math::Matrix::transform(vsInput.normal, transMat);
		psInput.uv = vsInput.uv;
		psInput.color = vsInput.color;
	}

	void pixelShader(const std::vector<Pixel>& pixels , std::vector<Color>& colors)
	{
		CallPixelShader(pixels ,  textures , colors);
	}
};