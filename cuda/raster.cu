/**
 * File Name : raster.h
 * Author : Yang Fan
 * Date : 2018/11/28
 * declare raster
 */

#pragma once

#include "../includes/math/vector.hpp"
#include "../common/define.h"
#include "../common/color.h"
#include <cmath>
#include <vector>

class Raster {
private:
	/*
	 * Interpolate float value
	 */
	static float Interpolate(float v1, float v2, float gad) {
		CLAMP01(gad);
		if (v1 > v2) { return v1 - (v1 - v2) * gad; }
		return v1 + (v2 - v1) * gad;
	}

	/*
	 * Interpolate vec3 value
	 */
	static Math::Vector3 Interpolate(const Math::Vector3& v1, const Math::Vector3& v2, float gad) {
		return Math::Vector3(Interpolate(v1._x, v2._x, gad), Interpolate(v1._y, v2._y, gad),
		                     Interpolate(v1._z, v2._z, gad));
	}

	/*
	 * Interpolate vec2 value
	 */
	static Math::Vector2 Interpolate(const Math::Vector2& v1, const Math::Vector2& v2, float gad) {
		return Math::Vector2(Interpolate(v1._x, v2._x, gad), Interpolate(v1._y, v2._y, gad));
	}

	/*
	 * Interpolate color value
	 */
	static Color Interpolate(const Color& v1, const Color& v2, float gad) {
		return Color(Interpolate(v1.r, v2.r, gad), Interpolate(v1.g, v2.g, gad), Interpolate(v1.b, v2.b, gad),
		             Interpolate(v1.a, v2.a, gad));
	}

	/*
	 * Interpolate pixel value
	 */
	static Pixel Interpolate(const Pixel& p1, const Pixel& p2, float gad) {
		Pixel p;
		p.pos = Interpolate(p1.pos, p2.pos, gad);
		p.normal = Interpolate(p1.normal, p2.normal, gad);
		p.uv = Interpolate(p1.uv, p2.uv, gad);
		p.color = Interpolate(p1.color, p2.color, gad);
		return p;
	}

	/*
	 * Bresenham Line Algorithm
	 */
	static void RasterizeLine(const Pixel& p1, const Pixel& p2, std::vector<Pixel>& pixels) {

		auto start = p1, end = p2;

		if (p1.pos._x > p2.pos._x) {
			start = p2;
			end = p1;
		}

		float gad = 0.0f;
		float disx = abs(end.pos._x - start.pos._x);
		float disy = abs(end.pos._y - start.pos._y);
		float dis = disx > disy ? disx : disy;

		for (auto i = 0; i < dis; i++) {
			gad = i / dis;
			pixels.push_back(Interpolate(p1, p2, gad));
		}
	}

	static void RasterizeTriangle(const Pixel& top, const Pixel& mid, const Pixel& btm, std::vector<Pixel>& pixels) {
		for (auto y = top.pos._y; y >= btm.pos._y; --y) {
			Pixel sp, ep;
			float sgad = 0.0f, egad = 0.0f;
			if (y >= mid.pos._y) {
				sgad = (y - top.pos._y) / (mid.pos._y - top.pos._y);
				egad = (y - top.pos._y) / (btm.pos._y - top.pos._y);
				sp = Interpolate(top, mid, sgad);
				ep = Interpolate(top, btm, egad);
			}
			else {
				sgad = (y - mid.pos._y) / (btm.pos._y - mid.pos._y);
				egad = (y - top.pos._y) / (btm.pos._y - top.pos._y);
				sp = Interpolate(mid, btm, sgad);
				ep = Interpolate(top, btm, egad);
			}


			if (sp.pos._x > ep.pos._x) {
				auto tp = sp;
				sp = ep;
				ep = tp;
			}

			float sx = sp.pos._x, ex = ep.pos._x; // x坐标	

			for (auto x = sx; x <= ex; ++x) {
				float gad = 0;
				if (ex - sx < 1.0f) { gad = 0; }
				else gad = (x - sx) / (ex - sx);
				pixels.push_back(Interpolate(sp, ep, gad));
			}
		}
	}

public:

	/*
	 * 定义临时数据结构
	 */
	struct Triangle {
		/*
		 * 三个顶点
		 */
		Pixel top, mid, btm;
		/*
		 * 像素个数
		 */
		int numPixels;
	};

	static void rasterize(Triangle& triangle, std::vector<Pixel>& pixels, const TYPE type) {
		if (type == SOLID) {
			RasterizeTriangle(triangle.top, triangle.mid, triangle.btm, pixels);
		}
		else {
			RasterizeLine(triangle.top, triangle.mid, pixels);
			RasterizeLine(triangle.top, triangle.btm, pixels);
			RasterizeLine(triangle.mid, triangle.btm, pixels);
		}
	}
};
