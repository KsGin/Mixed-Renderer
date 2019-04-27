/**
 * File Name : raster.h
 * Author : Yang Fan
 * Date : 2019/4/20
 * doRenderPipe pipe
 */


#pragma once
#include "define.h"
#include <vector>

extern "C" void CallRasterizeLines(const std::vector<Line>& lines, std::vector<Pixel>& pixels);

class Raster {

	static void doRasterizeSolidTriangle(Triangle& triangle) {
		auto top = triangle.top;
		auto mid = triangle.mid;
		auto btm = triangle.btm;

		for (auto y = top.pos._y; y >= btm.pos._y; --y) {
			Line tLine;

			auto sgad = 0.0f, egad = 0.0f;
			if (y >= mid.pos._y) {
				sgad = (y - top.pos._y) / (mid.pos._y - top.pos._y);
				egad = (y - top.pos._y) / (btm.pos._y - top.pos._y);
				INTERPOLATEP(top , mid , sgad , tLine.left);
				INTERPOLATEP(top , btm , egad , tLine.right);
			}
			else {
				sgad = (y - mid.pos._y) / (btm.pos._y - mid.pos._y);
				egad = (y - top.pos._y) / (btm.pos._y - top.pos._y);
				INTERPOLATEP(mid , btm , sgad , tLine.left);
				INTERPOLATEP(top , btm , egad , tLine.right);
			}


			if (tLine.left.pos._x > tLine.right.pos._x) {
				auto tp = tLine.left;
				tLine.left = tLine.right;
				tLine.right = tp;
			}

			tLine.numPixels = tLine.right.pos._x - tLine.left.pos._x + 1;
			if (tLine.numPixels >= 0) {
				lines.emplace_back(tLine);
			}
		}
	}

	static void doComputeTriangle(std::vector<Triangle>& triangles, const TYPE& type, int& numPixels) {
		for (auto& triangle : triangles) {
			// 背面剔除
			if (doCcwJudge(triangle)) continue;

			auto top = triangle.top;
			Device::FixedPoint(top.pos);
			auto mid = triangle.mid;
			Device::FixedPoint(mid.pos);
			auto btm = triangle.btm;
			Device::FixedPoint(btm.pos);

			// 修正三个点的位置 
			if (btm.pos._y > mid.pos._y) { std::swap(btm, mid); }
			if (mid.pos._y > top.pos._y) { std::swap(mid, top); }
			if (btm.pos._y > mid.pos._y) { std::swap(btm, mid); }

			triangle.top = top;
			triangle.mid = mid;
			triangle.btm = btm;

			if (type == SOLID) {
				triangle.numPixels = 0.5 * abs(
					(mid.pos._x - top.pos._x) * (btm.pos._y - top.pos._y) - (btm.pos._x - top.pos._x) * (mid.pos._y -
						top.pos._y));
			}
			else {
				triangle.numPixels = Distance(top.pos, mid.pos) + Distance(top.pos, btm.pos) + Distance(
					mid.pos, btm.pos);
			}
			numPixels += triangle.numPixels;
		}
	}
	
	static void doRasterizeWireframeTriangle(Triangle& triangle) {
		Line line1 , line2 , line3;
		line1.left = triangle.top;
		line1.right = triangle.btm;
		line1.numPixels = Distance(line1.left.pos , line1.right.pos);

		line2.left = triangle.top;
		line2.right = triangle.mid;
		line2.numPixels = Distance(line2.left.pos , line2.right.pos);

		line3.left = triangle.mid;
		line3.right = triangle.btm;
		line3.numPixels = Distance(line3.left.pos , line3.right.pos);

		if (line1.left.pos._x > line1.right.pos._x) std::swap(line1.left , line1.right);
		if (line2.left.pos._x > line2.right.pos._x) std::swap(line2.left , line2.right);
		if (line3.left.pos._x > line3.right.pos._x) std::swap(line3.left , line3.right);

		lines.emplace_back(line1);
		lines.emplace_back(line2);
		lines.emplace_back(line3);
	}

	static bool doCcwJudge(const Triangle& triangle) {
		auto d1 = triangle.top.pos - triangle.mid.pos;
		auto d2 = triangle.btm.pos - triangle.mid.pos;
		d1._z = 0;
		d2._z = 0;

		const auto d = Math::Vector3::cross(d1 , d2).normalize();
		const auto lhr = Math::Vector3(0 , 0 , 1);

		return Math::Vector3::dot(d , lhr) < 0;
	}

	static void doReset() {
		lines.clear();
		lines.shrink_to_fit();
	}

public:

	static std::vector<Line> lines;

	static void doRasterize(std::vector<Triangle>& triangles , std::vector<Pixel>& pixels ,const TYPE& type = SOLID ) {
		
		doReset();
		
		if (type == SOLID) {
			auto numPixels = 0;
			doComputeTriangle(triangles, type , numPixels);

			if (numPixels >= pixels.size()) {
				pixels.resize(numPixels * 1.5);
			}
			
			for (auto& triangle : triangles) {				
				doRasterizeSolidTriangle(triangle );
			}
				
			CallRasterizeLines(lines , pixels);

		} else {
			auto numPixels = 0;
			doComputeTriangle(triangles, type , numPixels);

			if (numPixels >= pixels.size()) {
				pixels.resize(numPixels * 1.5);
			}

			for (auto& triangle : triangles) {
				doRasterizeWireframeTriangle(triangle);
			}	

			CallRasterizeLines(lines , pixels);
		}
	}
};


std::vector<Line> Raster::lines = std::vector<Line>(2560);