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
 * 摄像机类基类
 */
class Camera {
public:
	/*
	 * 近平面
	 */
	float near;
	/*
	 * 远平面
	 */
	float far;

	/*
	 * 缩放
	 */
	float fovScale;

	/*
	 * eye 坐标
	 */
	Math::Vector3 eye;

	/*
	 * 观察点
	 */
	Math::Vector3 target;

	/*
	 * 构造方法
	 */
	Camera() {
		near = 0;
		far = 0;
		fovScale = 0;
		eye = Math::Vector3(0 , 0 , 0);
		target = Math::Vector3(0 , 0 , 0);
	}

	/*
	 * 空构造方法
	 */
	~Camera() {
		
	}

	/*
	 * 定义虚函数
	 */
	Ray generateRay(float x, float y) {
		
	};
};

/*
 * 透视投影摄像机
 */
class PerspectiveCamera : public Camera {
private:
	/*
	 * 方向向量 right
	 */
	Math::Vector3 right;

	/*
	 * 方向向量 up
	 */
	Math::Vector3 up;

	/*
	 * 方向向量 front
	 */
	Math::Vector3 front;

	/*
	 * 空构造方法
	 */
	PerspectiveCamera() {
		
	}

public:
	/*
	 * 构造方法
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
	 * 析构方法
	 */
	~PerspectiveCamera() {

	}

	/*
	 * 根据 x y 位置生成光线
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
 * 正交投影摄像机
 */
class OrthoCamera : public Camera {

};
