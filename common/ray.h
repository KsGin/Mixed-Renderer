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
	 * �Ƿ񼤻�
	 */
	bool isActive;

	/*
	 * �չ��췽��
	 */
	__device__ __host__ Ray() {
		this->origin = Math::Vector3(0, 0, 0);
		this->direction = Math::Vector3(0, 0, -1);
		this->isActive = false;
	}

	/*
	 * ֵ�������췽��
	 */
	__device__ __host__ Ray(const Math::Vector3& origin, const Math::Vector3& direction) {
		this->origin = origin;
		this->direction = direction;
		this->isActive = true;
	}

	/*
	 * ���󿽱����췽��
	 */
	__device__ __host__ Ray(const Ray& r) {
		this->origin = r.origin;
		this->direction = r.direction;
		this->isActive = r.isActive;
	}

	/*
	 * ���� =
	 */
	Ray& operator=(const Ray& r) {
		this->direction = r.direction;
		this->origin = r.origin;
		this->isActive = r.isActive;
		return *this;
	}

	__device__ __host__ void active() {
		this->isActive = true;
	}
};
