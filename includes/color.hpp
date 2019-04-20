/**
 * File Name : color.hpp
 * Author : Yang Fan
 * Date : 2018/11/27
 * color class
 */

#pragma once

#include <SDL_system.h>
#include <cuda_runtime.h>


/*
 * 颜色类
 */
class Color {
public:
	/*
	 * 数据定义
	 */
	union {
		struct {
			float r, g, b, a;
		};

		struct {
			float x, y, z, w;
		};
	};

	/*
	 * 空构造方法
	 */
	__device__ __host__ Color() {
		this->x = 0;
		this->y = 0;
		this->z = 0;
		this->w = 0;
	}

	/*
	 * 值构造方法
	 */
	__device__ __host__  Color(float r, float g, float b, float a) {
		this->x = r;
		this->y = g;
		this->z = b;
		this->w = a;
	}

	/*
	 * 引用拷贝
	 */
	__device__ __host__  Color(const Color& color) {
		this->r = color.r;
		this->g = color.g;
		this->b = color.b;
		this->a = color.a;
	}

	/*
	 * 重载 =
	 */
	__device__ __host__  Color& operator=(const Color& color) {
		this->r = color.r;
		this->g = color.g;
		this->b = color.b;
		this->a = color.a;
		return *this;
	}

	/*
	 * 重载 +
	 */
	__device__ __host__  Color operator+(const Color& color) { return Color(r + color.r, g + color.g, b + color.b, a + color.a); }

	/*
	 * 重载 -
	 */
	__device__ __host__  Color operator-(const Color& color) { return Color(r - color.r, g - color.g, b - color.b, a - color.a); }

	/*
	 * 重载 *
	 */
	__device__ __host__  Color operator*(const Color& color) { return Color(r * color.r, g * color.g, b * color.b, a * color.a); }

	/*
	 * 重载 /
	 */
	__device__ __host__  Color operator/(const Color& color) { return Color(r / color.r, g / color.g, b / color.b, a / color.a); }

	/*
	 * 重载 +
	 */
	__device__ __host__  Color operator+(const float c) { return Color(r + c, g + c, b + c, a + c); }

	/*
	 * 重载 -
	 */
	__device__ __host__  Color operator-(const float c) { return Color(r - c, g - c, b - c, a - c); }

	/*
	 * 重载 *
	 */
	__device__ __host__  Color operator*(const float c) { return Color(r * c, g * c, b * c, a * c); }

	/*
	 * 重载 /
	 */
	__device__ __host__  Color operator/(const float c) { return Color(r / c, g / c, b / c, a / c); }

	/*
	 * 限制颜色在 0 - 1 之间
	 */
	__device__ __host__  Color& modulate() {
		this->r = (r > 0 ? r : 0) < 1 ? r : 1;
		this->g = (g > 0 ? g : 0) < 1 ? g : 1;
		this->b = (b > 0 ? b : 0) < 1 ? b : 1;
		this->a = (a > 0 ? a : 0) < 1 ? a : 1;
		return *this;
	}

	/*
	 * 红色
	 */
	__device__ __host__  static Color red() { return Color(1, 0, 0, 1); }

	/*
	 * 绿色
	 */
	__device__ __host__  static Color green() { return Color(0, 1, 0, 1); }

	/*
	 * 蓝色
	 */
	__device__ __host__  static Color blue() { return Color(0, 0, 1, 0); }

	/*
	 * 白色
	 */
	__device__ __host__  static Color white() { return Color(1, 1, 1, 1); }

	/*
	 * 黑色
	 */
	__device__ __host__  static Color black() { return Color(0, 0, 0, 0); }
};

