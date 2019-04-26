/**
 * File Name : ray.h
 * Author : Yang Fan
 * Date : 2018/4/26
 * define ray class
 */

#pragma once

#include "../includes/math/vector.hpp"

/*
 * ������
 */
class Ray {

public:
	/*
	 * ������� origin
	 */
	Math::Vector3 origin;
	/*
	 * ���߷��� direction
	 */
	Math::Vector3 direction;

	/*
	 * �չ��췽��
	 */
	Ray() {
		this->origin = Math::Vector3(0, 0, 0);
		this->direction = Math::Vector3(0, 0, -1);
	}

	/*
	 * ֵ�������췽��
	 */
	Ray(const Math::Vector3& origin, const Math::Vector3& direction) {
		this->origin = origin;
		this->direction = direction;
	}

	/*
	 * ���󿽱����췽��
	 */
	Ray(const Ray& r) {
		this->origin = r.origin;
		this->direction = r.direction;
	}

	/*
	 * ���� =
	 */
	Ray& operator=(const Ray& r) {
		this->direction = r.direction;
		this->origin = r.origin;
		return *this;
	}
};
