/**
 * File Name : shader.h
 * Author : Yang Fan
 * Date : 2019/1/7
 * defined shader
 */

#pragma once

#include "../includes/math/matrix.hpp"
#include "define.h"
#include "../common/texture.h"
#include <vector>
#include "../includes/color.hpp"

extern "C" void CallPixelShader(const std::vector<Pixel>& pixels,const std::vector<Texture>& textures , const SHADER_TYPE& sType , std::vector<Color>& colors , const Args& args);

class Shader
{
	/*
	 * shader type
	 */
	SHADER_TYPE sType;
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
	/*
	 * args
	 */
	Args args;
public:
	/*
	* Constructor
	*/
	Shader(const SHADER_TYPE& shaderType = CUBE)
	{
		sType = shaderType;
	}

	/*
	 * Deconstructor
	 */
	~Shader()
	{
	}

	void setArgs(const Args& shaderArgs) {
		args = shaderArgs;
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
	void vertexShader(const Vertex& vsInput, Pixel& psInput) const
	{
		auto transMat = modelMat.multiply(viewMat).multiply(perspectiveMat);
		psInput.pos = Math::Matrix::transformCoordinates(vsInput.pos, transMat);
		psInput.normal = Math::Matrix::transform(vsInput.normal, transMat);
		psInput.uv = vsInput.uv;
		psInput.color = vsInput.color;
	}

	void pixelShader(const std::vector<Pixel>& pixels , std::vector<Color>& colors) const
	{
		if (pixels.size() >= colors.size()) {
			colors.resize(pixels.size());
		}

		CallPixelShader(pixels , textures , sType , colors , args);
	}
};