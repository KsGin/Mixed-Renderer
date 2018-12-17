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

#include <iostream>

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

	void ProcessLineDrawTriangle(const Math::Vector2& p1, const Math::Vector2& p2, const Math::Vector2& p3,
	                             const Color& color) const {
		Math::Vector2 top = p1, mid = p2, btm = p3, tmp;

		// 修正三个点的位置 
		if (btm._y > mid._y) {
			tmp = mid;
			mid = btm;
			btm = tmp;
		}

		if (mid._y > top._y) {
			tmp = top;
			top = mid;
			mid = tmp;
		}

		if (btm._y > mid._y) {
			tmp = mid;
			mid = btm;
			btm = tmp;
		}

		// printf("top => (%.2f,%.2f) , mid => (%.2f,%.2f) , btm => (%.2f,%.2f)\n" , top._x , top._y , mid._x , mid._y , btm._x , btm._y);

		// 三角形顶部点和其他两个点的反向斜率
		float dtm = 0, dtb = 0, dmb = 0;
		if (top._y - mid._y > 0) { dtm = (top._x - mid._x) / (top._y - mid._y); }
		if (top._y - btm._y > 0) { dtb = (top._x - btm._x) / (top._y - btm._y); }
		if (mid._y - btm._y > 0) { dmb = (mid._x - btm._x) / (mid._y - btm._y); }

		// printf("%.2f  %.2f\n", dtm, dtb);

		for (auto y = top._y; y >= btm._y; --y) {
			float sx = 0, ex = 0; // x坐标
			if (y > mid._y) {
				sx = top._x + dtm * (y - top._y);
				ex = top._x + dtb * (y - top._y);
			}
			else {
				sx = mid._x + dmb * (y - mid._y);
				ex = top._x + dtb * (y - top._y);
			}

			if (sx > ex) {
				const float t = sx;
				sx = ex;
				ex = t;
			}

			for (auto x = sx; x <= ex; ++x) { device->setPixelColor(x, y, color); }
		}
	}

	Math::Vector2 FixedPoint2D(const Math::Vector2& p) {
		Math::Vector2 refP;
		refP._x =  p._x * device->width + device->width / 2;
		refP._y = -p._y * device->height + device->height / 2;
		return refP;
	}
public:

	/*
	 * 绘制类型
	 */
	enum TYPE {
		SOLID ,
		WIREFRAME
	};

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

	void draw(const Math::Vector2& p1, const Math::Vector2& p2, const Math::Vector2& p3, const Color& color , const TYPE type) {

		const auto pd1 = FixedPoint2D(p1);
		const auto pd2 = FixedPoint2D(p2);
		const auto pd3 = FixedPoint2D(p3);

		if (type == SOLID) {
			ProcessLineDrawTriangle(pd1, pd2, pd3, color);
		} else {
			BresenhamRasterLine(pd1, pd2, color);
			BresenhamRasterLine(pd1, pd3, color);
			BresenhamRasterLine(pd2, pd3, color);
		}
	}

	
};
