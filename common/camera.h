/**
 * File Name : camera.h
 * Author : Yang Fan
 * Date : 2018/11/27
 * declare camera property and method
 */

#pragma once

#include "define.h"
#include "../includes/math/vector.hpp"
#include "ray.h"

/*
 * ͸��ͶӰ�����
 */
class PerspectiveCamera{
public:
	/*
	 * ��ƽ��
	 */
	float near;
	/*
	 * Զƽ��
	 */
	float far;

	/*
	 * ����
	 */
	float fovScale;

	/*
	 * eye ����
	 */
	Math::Vector3 eye;

	/*
	 * �۲��
	 */
	Math::Vector3 target;

private:
	/*
	 * �������� right
	 */
	Math::Vector3 right;

	/*
	 * �������� up
	 */
	Math::Vector3 up;

	/*
	 * �������� front
	 */
	Math::Vector3 front;

public:
	/*
	 * �չ��췽��
	 */
	__device__ __host__ PerspectiveCamera(): near(0), far(0), fovScale(0) {

	}

	/*
	 * ���췽��
	 */
	__device__ __host__ PerspectiveCamera(float fov, const Math::Vector3& eye, const Math::Vector3& target, const Math::Vector3& up,
	                  float near, float far) {
		this->near = near;
		this->far = far;
		this->eye = eye;
		this->target = target;
		this->front = (target - this->eye).normalize();
		this->right = Math::Vector3::cross(front, up).normalize();
		this->up = Math::Vector3::cross(right, front).normalize();
		this->fovScale = static_cast<float>(tan(fov * 0.5 * M_PI / 180) * 2);
	}

	/*
	 * ���� x y λ�����ɹ���
	 */
	__device__ __host__ void generateRay(const float& x, const float& y , Ray &ray) const{

		auto scaleX = static_cast<float>((x - 0.5) * this->fovScale);
		auto r = this->right * scaleX;

		auto scaleY = static_cast<float>((y - 0.5) * this->fovScale);
		auto u = this->up * scaleY;

		ray.origin = eye;
		ray.direction = (front + r + u).normalize();

		ray.active();
	}
};

/*
 * ����ͶӰ�����
 */
class OrthoCamera{

};
