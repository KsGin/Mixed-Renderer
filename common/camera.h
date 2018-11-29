/**
 * File Name : camera.cu
 * Author : Yang Fan
 * Date : 2018/11/27
 * declare camera property and method
 */

#pragma once

#include "../includes/math/vector.hpp"

class Camera
{
	/**
	 * camera position
	 */
	Math::Vector3 position;

	/**
	 * camera look at 
	 */
	Math::Vector3 target;

	/**
	 * near
	 */
	float near;

	/**
	 * far
	 */
	float far;

protected:
	/**
	 * private camera
	 */
	Camera()
	{
		position = Math::Vector3();
		target = Math::Vector3();
		far = 0;
		near = 0;
	}

	/**
	 * public ~camera
	 */
	~Camera()
	{
		// do something
	}
};

class OrthographicCamera : Camera
{
	/*
	 * view width
	 */
	float width;

	/*
	 * view height
	 */
	float height;

public:

	OrthographicCamera() : Camera()
	{
		width = 0;
		height = 0;
	}
};

class PerspectiveCamera : Camera
{
	/*
	 * view fov
	 */
	float fov;

public:
	PerspectiveCamera() : Camera()
	{
		fov = 0;
	}
};