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
 * ����������
 */
class Camera {
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

	/*
	 * ���췽��
	 */
	Camera() {
		near = 0;
		far = 0;
		fovScale = 0;
		eye = Math::Vector3(0 , 0 , 0);
		target = Math::Vector3(0 , 0 , 0);
	}

	/*
	 * �չ��췽��
	 */
	~Camera() {
		
	}

	/*
	 * �����麯��
	 */
	Ray generateRay(float x, float y) {
		
	};
};

/*
 * ͸��ͶӰ�����
 */
class PerspectiveCamera : public Camera {
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

	/*
	 * �չ��췽��
	 */
	PerspectiveCamera() {
		
	}

public:
	/*
	 * ���췽��
	 */
	PerspectiveCamera(float fov, const Math::Vector3& eye, const Math::Vector3& target, const Math::Vector3& up,
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
	 * ��������
	 */
	~PerspectiveCamera() {

	}

	/*
	 * ���� x y λ�����ɹ���
	 */
	Ray generateRay(float x, float y) {

		auto scaleX = static_cast<float>((x - 0.5) * this->fovScale);
		auto r = this->right * scaleX;

		auto scaleY = static_cast<float>((y - 0.5) * this->fovScale);
		auto u = this->up * scaleY;

		return Ray(eye, (front + r + u).normalize());
	}
};

/*
 * ����ͶӰ�����
 */
class OrthoCamera : public Camera {

};
