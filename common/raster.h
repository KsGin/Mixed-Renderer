/**
 * File Name : raster.cu
 * Author : Yang Fan
 * Date : 2018/11/28
 * declare raster
 */

#pragma once

#include "../includes/math/vector.hpp"
#include "color.h"
#include "device.h"
#include <cmath>

class Raster {
private:

	/*
	 * DEVICE CONTEXT
	 */
	Device* device;

	/*
	 * Interpolate float value
	 */
	float Interpolate(float v1, float v2, float gad) const {
		if (v1 > v2) { return v1 - (v1 - v2) * device->clamp(gad); }
		return v1 + (v2 - v1) * device->clamp(gad);
	}

	/*
	 * Interpolate vec3 value
	 */
	Math::Vector3 Interpolate(const Math::Vector3& v1, const Math::Vector3& v2, float gad) const {
		return Math::Vector3(Interpolate(v1._x, v2._x, gad), Interpolate(v2._y, v2._y, gad), Interpolate(v1._z, v2._z, gad));
	}

	/*
	 * Interpolate vec2 value
	 */
	Math::Vector2 Interpolate(const Math::Vector2& v1, const Math::Vector2& v2, float gad) const {
		return Math::Vector2(Interpolate(v1._x, v2._x, gad), Interpolate(v2._y, v2._y, gad));
	}

	/*
	 * Bresenham Line Algorithm
	 */
	void BresenhamRasterLine(const Math::Vector2& p1, const Math::Vector2& p2, const Color& color) {
		int x1 = p1._x, x2 = p2._x, y1 = p1._y, y2 = p2._y;
		const int dx = abs(x2 - x1);
		const int dy = abs(y2 - y1);
		int sx = x1 < x2 ? 1 : -1;
		int sy = y1 < y2 ? 1 : -1;
		int eps = dx - dy;
		while (true) {
			device->setPixelColor(x1, y1, color);
			if (x1 == x2 && y1 == y2) break;
			auto e2 = 2 * eps;
			if (e2 > -dy) {
				eps -= dy;
				x1 += sx;
			}

			if (e2 < dx) {
				eps += dx;
				y1 += sy;
			}
		}
	}

public:
	/**
	 * 构造方法
	 */
	Raster() {
	}

	/*
	 * 析构方法
	 */
	~Raster() {
	}

	void initialize(Device* device) { this->device = device; }

	void drawLine(const Math::Vector2& p1, const Math::Vector2& p2, const Color& color) {
		BresenhamRasterLine(p1, p2, color);
	}
};
