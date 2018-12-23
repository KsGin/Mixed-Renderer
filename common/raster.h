/**
 * File Name : raster.h
 * Author : Yang Fan
 * Date : 2018/11/28
 * declare raster
 */

#pragma once

#include "../includes/math/vector.hpp"
#include "color.h"
#include "device.h"
#include "model.h"
#include "shader.h"
#include <cmath>

#include <vector>
#include <iostream>

static class Raster {
private:

	/*
	 * Distance of two points
	 */
	static int Distance(const Math::Vector3& p1, const Math::Vector3& p2) {
		return sqrt(pow(p2._x - p1._x , 2) + pow(p2._y - p1._y , 2));
	}

	/*
	 * Fixed point function
	 */
	static Math::Vector3 FixedPoint2D(const Math::Vector3& p) {
		Math::Vector3 refP;
		refP._x = p._x * Device::getInstance().width + Device::getInstance().width / 2;
		refP._y = -p._y * Device::getInstance().height + Device::getInstance().height / 2;
		refP._z = p._z;
		return refP;
	}

	/*
	 * Interpolate float value
	 */
	static float Interpolate(float v1, float v2, float gad){
		if (v1 > v2) { return v1 - (v1 - v2) * Device::clamp(gad); }
		return v1 + (v2 - v1) * Device::clamp(gad);
	}

	/*
	 * Interpolate vec3 value
	 */
	static Math::Vector3 Interpolate(const Math::Vector3& v1, const Math::Vector3& v2, float gad){
		return Math::Vector3(Interpolate(v1._x, v2._x, gad), Interpolate(v1._y, v2._y, gad), Interpolate(v1._z, v2._z, gad));
	}

	/*
	 * Interpolate vec2 value
	 */
	static Math::Vector2 Interpolate(const Math::Vector2& v1, const Math::Vector2& v2, float gad){
		return Math::Vector2(Interpolate(v1._x, v2._x, gad), Interpolate(v2._y, v2._y, gad));
	}

	/*
	 * Interpolate color value
	 */
	static Color Interpolate(const Color& v1, const Color& v2, float gad) {
		return Color(Interpolate(v1.r, v2.r, gad), Interpolate(v1.g, v2.g, gad), Interpolate(v1.b, v2.b, gad), Interpolate(v1.a, v2.a, gad));
	}

	/*
	 * Interpolate pixel value
	 */
	static Shader::PSInput Interpolate(const Shader::PSInput& p1, const Shader::PSInput& p2, float gad) {
		Shader::PSInput p;
		p.pos = Interpolate(p1.pos, p2.pos, gad);
		p.normal = Interpolate(p1.normal, p2.normal, gad);
		p.color = Interpolate(p1.color, p2.color, gad);
		return p;
	}

	/*
	 * Bresenham Line Algorithm
	 */
	static void RasterizeLine(const Shader::PSInput& p1, const Shader::PSInput& p2, std::vector<Shader::PSInput> &pixels , size_t &idx) {

		auto start = p1, end = p2;

		if (p1.pos._x > p2.pos._x) { start = p2; end = p1; }

		float gad = 0.0f;
		float disx = abs(end.pos._x - start.pos._x);
		float disy = abs(end.pos._y - start.pos._y);
		float dis = disx > disy ? disx : disy;

		for (auto i = 0; i < dis; i++)
		{
			gad = i / dis;
			if (idx >= pixels.capacity()) pixels.resize(idx * 1.5);
			pixels[idx++] = Interpolate(p1, p2, gad);
		}
	}

	static void RasterizeTriangle(const Shader::PSInput& top, const Shader::PSInput& mid, const Shader::PSInput& btm, std::vector<Shader::PSInput> &pixels , size_t &idx){

		// 三角形顶部点和其他两个点的反向斜率
		float dtm = 0, dtb = 0, dmb = 0;
		if (top.pos._y - mid.pos._y > 0) { dtm = (top.pos._x - mid.pos._x) / (top.pos._y - mid.pos._y); }
		if (top.pos._y - btm.pos._y > 0) { dtb = (top.pos._x - btm.pos._x) / (top.pos._y - btm.pos._y); }
		if (mid.pos._y - btm.pos._y > 0) { dmb = (mid.pos._x - btm.pos._x) / (mid.pos._y - btm.pos._y); }

		for (auto y = top.pos._y; y >= btm.pos._y; --y) {
			Shader::PSInput sp, ep;
			float sgad = 0.0f , egad = 0.0f;
			if (y > mid.pos._y) {
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


			if (sp.pos._x > ep.pos._x)
			{
				auto tp = sp;
				sp = ep;
				ep = tp;
			}

			float sx = sp.pos._x, ex = ep.pos._x;  // x坐标	

			for (auto x = sx; x <= ex; ++x) { 
				float gad = (x - sx) / (ex - sx);
				if (idx >= pixels.capacity()) 
					pixels.resize(idx * 1.5);
				pixels[idx++] = Interpolate(sp, ep, gad);
			}
		}
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
	static void rasterize(const Shader::PSInput& p1 , const Shader::PSInput& p2 , const Shader::PSInput& p3 , std::vector<Shader::PSInput>& pixels , const TYPE type) {

		auto top = p1; top.pos = FixedPoint2D(top.pos);
		auto mid = p2; mid.pos = FixedPoint2D(mid.pos);
		auto btm = p3; btm.pos = FixedPoint2D(btm.pos);

		Shader::PSInput tmp;
		// 修正三个点的位置 
		if (btm.pos._y > mid.pos._y) {
			tmp = mid;
			mid = btm;
			btm = tmp;
		}

		if (mid.pos._y > top.pos._y) {
			tmp = top;
			top = mid;
			mid = tmp;
		}

		if (btm.pos._y > mid.pos._y) {
			tmp = mid;
			mid = btm;
			btm = tmp;
		}

		size_t numPixels = 0 , idx = 0;
		size_t tm = Distance(top.pos, mid.pos), tb = Distance(top.pos, btm.pos), mb = Distance(mid.pos, btm.pos);

		if (type == SOLID) {
			size_t p = (tm + tb + mb) / 2;
			numPixels = sqrt(p * (p - tm) * (p - tb) * (p - mb));
			pixels.resize(numPixels * 1.5);

			RasterizeTriangle(top, mid, btm, pixels , idx);
		} else {
			numPixels = tm + tb + mb;
			pixels.resize(numPixels * 1.5);

			RasterizeLine(top, mid, pixels , idx);
			RasterizeLine(top, btm, pixels , idx);
			RasterizeLine(mid, btm, pixels , idx);
		}
	}
};
