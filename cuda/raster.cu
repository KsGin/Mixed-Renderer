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

/*
 * INTERPOLATE FLOAT VALUE DEFINED
 */
#define INTERPOLATE(a , b , g , r) {								\
	CLAMP01(g);													\
	int d = a > b;													\
	r = d * (a - (a - b) * g) + (1-d) * (a + (b - a) * g);		\
}

/*
 * INTERPOLATE VECTOR2 VALUE DEFINED
 */
#define INTERPOLATEV2(v1 , v2 , gad , result) {								\
	INTERPOLATE(v1._x, v2._x, gad , result._x);								\
	INTERPOLATE(v1._y, v2._y, gad , result._y);								\
}

/*
 * INTERPOLATE VECTOR3 VALUE DEFINED
 */
#define INTERPOLATEV3(v1 , v2 , gad , result) {								\
	INTERPOLATE(v1._x, v2._x, gad , result._x);								\
	INTERPOLATE(v1._y, v2._y, gad , result._y);								\
	INTERPOLATE(v1._z, v2._z, gad , result._z);								\
}

/*
 * INTERPOLATE COLOR VALUE DEFINED
 */
#define INTERPOLATEC(v1 , v2 , gad , result) {								\
	INTERPOLATE(v1.r, v2.r, gad , result.r);								\
	INTERPOLATE(v1.g, v2.g, gad , result.g);								\
	INTERPOLATE(v1.b, v2.b, gad , result.b);								\
	INTERPOLATE(v1.a, v2.a, gad , result.a);								\
}

#define INTERPOLATEP(p1 , p2 , gad , result) {								\
	INTERPOLATEV3(p1.pos , p2.pos , gad , result.pos);						\
	INTERPOLATEV3(p1.normal , p2.normal , gad , result.normal);				\
	INTERPOLATEV2(p1.uv , p2.uv , gad , result.uv);							\
	INTERPOLATEC(p1.color , p2.color , gad , result.color);					\
}

class Raster {
private:
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

		Pixel p;
		for (auto i = 0; i < dis; i++) {
			gad = i / dis;
			INTERPOLATEP(p1 , p2 , gad , p);
			pixels.emplace_back(p);
		}
	}

	static void RasterizeTriangle(const Pixel& top, const Pixel& mid, const Pixel& btm, std::vector<Pixel>& pixels) {
		for (auto y = top.pos._y; y >= btm.pos._y; --y) {
			Pixel sp, ep;
			float sgad = 0.0f, egad = 0.0f;
			if (y >= mid.pos._y) {
				sgad = (y - top.pos._y) / (mid.pos._y - top.pos._y);
				egad = (y - top.pos._y) / (btm.pos._y - top.pos._y);
				INTERPOLATEP(top , mid , sgad , sp);
				INTERPOLATEP(top , btm , egad , ep);
			}
			else {
				sgad = (y - mid.pos._y) / (btm.pos._y - mid.pos._y);
				egad = (y - top.pos._y) / (btm.pos._y - top.pos._y);
				INTERPOLATEP(mid , btm , sgad , sp);
				INTERPOLATEP(top , btm , egad , ep);
			}


			if (sp.pos._x > ep.pos._x) {
				auto tp = sp;
				sp = ep;
				ep = tp;
			}

			float sx = sp.pos._x, ex = ep.pos._x; // x坐标	

			Pixel p;
			for (auto x = sx; x <= ex; ++x) {
				float gad = 0;
				if (ex - sx < 1.0f) { gad = 0; }
				else gad = (x - sx) / (ex - sx);
				INTERPOLATEP(sp , ep , gad , p);
				pixels.emplace_back(p);
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
