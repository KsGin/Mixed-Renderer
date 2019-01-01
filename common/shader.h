/**
 * File Name : shader.h
 * Author : Yang Fan
 * Date : 2018/12/19
 * declare pipe shader
 */
#pragma once

#include "../includes/math/matrix.hpp"
#include "../includes/math/vector.hpp"
#include "texture.h"
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
	PSInput vertexShader(const VSInput& vsInput){
		Math::Matrix transMat = modelMat.multiply(viewMat).multiply(perspectiveMat);
		PSInput psInput;
		psInput.pos = Math::Matrix::transformCoordinates(vsInput.pos , transMat);
		psInput.normal = Math::Matrix::transform(vsInput.normal, transMat);
		psInput.uv = vsInput.uv;
		psInput.color = vsInput.color;
		return psInput;
	}

	/*
	 * Pixel Shader
	 */
	Color pixelShader(const PSInput& psInput){
		Color color;
		color = textures[0].getPixel(psInput.uv._x , psInput.uv._y);
		return color;
	}
};
