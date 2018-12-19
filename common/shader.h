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
	* Raster 
	*/
	Raster *r;

	/*
	* Instance
	*/
	Shader *instance;

	/*
	* Vertex Transform
	*/
	void transform(const Math::Matrix &mat) {

	}

	/*
	* Vertex Shader
	*/
	void vs() {

	}

	/*
	* Pixel Shader
	*/
	void ps() {

	}

	/*
	* Disabled Constructor
	*/
	Shader() {
		r = new Raster();
	}

	/*
	* Disabled Deconstructor
	*/
	~Shader() {
		if (r) {
			delete r;
			r = nullptr;
		}
	}

public:

	/*
	* Singleton defined
	*/
	Shader* getInstance() {
		if (instance == nullptr)
		{
			instance = new Shader();
		}
		return instance;
	}

	/*
	* Render interface
	*/
	void render(const Model &model , const Math::Matrix &mat , const Device& immediateDevice) {
		transform(mat);

		vs();

		ps();
	}
};
