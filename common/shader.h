/**
 * File Name : shader.h
 * Author : Yang Fan
 * Date : 2018/12/19
 * declare pipe shader
 */
#pragma once

#include "../includes/math/matrix.hpp"
#include "../includes/math/vector.hpp"
#include "device.h"
#include "raster.h"
#include "model.h"

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
		Color color;
	};

	struct PSInput
	{
		Math::Vector3 pos;
		Math::Vector3 normal;
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
	 * Vertex Shader
	 */
	PSInput vertexShader(const VSInput& vsInput) const{
		Math::Matrix transMat = modelMat.multiply(viewMat).multiply(perspectiveMat);
		PSInput psInput;
		psInput.pos = Math::Matrix::transform(vsInput.pos , transMat);
		psInput.normal = Math::Matrix::transformCoordinates(vsInput.normal, transMat);
		psInput.color = vsInput.color;
		return psInput;
	}

	/*
	 * Pixel Shader
	 */
	Color pixelShader(const PSInput& psInput) const{
		Color color;
		color = psInput.color;
		return color;
	}
};
