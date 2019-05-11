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

extern "C" void CallPixelShader(std::unordered_map<ShaderType , ShaderData>& sDataMaps , const std::vector<Pixel>& pixels, std::vector<Color>& colors);

class Shader
{
public:

	ShaderData sData;

	/*
	* Constructor
	*/
	Shader(const ShaderType& shaderType = CUBE)
	{
		sData.sType = shaderType;
	}

	/*
	 * Deconstructor
	 */
	~Shader()
	{
	}

	void setArgs(const Args& shaderArgs) {
		sData.args = shaderArgs;
	}

	/*
	 * Set Matrix
	 */
	void setMat(const Math::Matrix& mat, const MatType& type)
	{
		switch (type)
		{
		case MODEL: sData.modelMat = mat;
			break;
		case VIEW: sData.viewMat = mat;
			break;
		case PERSPECTIVE: sData.perspectiveMat = mat;
			break;
		}
	}

	/*
	 * Set Texture
	 */
	void setTexture(const Texture& texture, const int idx)
	{
		this->sData.textures[idx] = texture;
	}

	
};


/*
 * Vertex Shader
 */
void vertexShader(std::unordered_map<ShaderType , ShaderData>& sDataMaps , const Vertex& vsInput, Pixel& psInput , ShaderType sType)
{
	const auto sData = sDataMaps[sType]; 
	const auto transMat = sData.modelMat.multiply(sData.viewMat).multiply(sData.perspectiveMat);
	const auto trans3DMat = sData.modelMat;
	psInput.pos = Math::Matrix::transformCoordinates(vsInput.pos, transMat);
	psInput.pos3D = Math::Matrix::transformCoordinates(vsInput.pos, trans3DMat);
	psInput.normal = Math::Matrix::transform(vsInput.normal, transMat);
	psInput.uv = vsInput.uv;
	psInput.color = vsInput.color;
}

void pixelShader(std::unordered_map<ShaderType , ShaderData>& sDataMaps , const std::vector<Pixel>& pixels , std::vector<Color>& colors) {
	if (pixels.size() >= colors.size()) {
		colors.resize(pixels.size());
	}

	CallPixelShader(sDataMaps , pixels , colors);
}