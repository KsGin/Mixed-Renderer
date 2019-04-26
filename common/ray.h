/**
 * File Name : ray.h
 * Author : Yang Fan
 * Date : 2018/4/26
 * define ray class
 */

#pragma once

#include "../includes/math/vector.hpp"

/*
 * 光线类
 */
class Ray {

public:
	/*
	 * 光线起点 origin
	 */
	Math::Vector3 origin;
	/*
	 * 光线方向 direction
	 */
	Math::Vector3 direction;

	/*
	 * 空构造方法
	 */
	Ray() {
		this->origin = Math::Vector3(0, 0, 0);
		this->direction = Math::Vector3(0, 0, -1);
	}

	/*
	 * 值拷贝构造方法
	 */
	Ray(const Math::Vector3& origin, const Math::Vector3& direction) {
		this->origin = origin;
		this->direction = direction;
	}

	/*
	 * 对象拷贝构造方法
	 */
	Ray(const Ray& r) {
		this->origin = r.origin;
		this->direction = r.direction;
	}

	/*
	 * 重载 =
	 */
	Ray& operator=(const Ray& r) {
		this->direction = r.direction;
		this->origin = r.origin;
		return *this;
	}
};
