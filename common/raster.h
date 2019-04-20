/**
 * File Name : raster.h
 * Author : Yang Fan
 * Date : 2019/4/20
 * render pipe
 */


#pragma once
#include "define.h"
#include <vector>

extern "C" void CallRasterizeLine(const Pixel& p1, const Pixel& p2, std::vector<Pixel>& pixels, size_t& index);
extern "C" void CallRasterizeTriangle(const Pixel& top, const Pixel& mid, const Pixel& btm, std::vector<Pixel>& pixels,size_t& index);

class Raster {
public:
	static void doRasterize(Triangle& triangle, std::vector<Pixel>& pixels, size_t& index, const TYPE type) {
		if (type == SOLID) {
			CallRasterizeTriangle(triangle.top, triangle.mid, triangle.btm, pixels, index);
		}
		else {
			CallRasterizeLine(triangle.top, triangle.mid, pixels, index);
			CallRasterizeLine(triangle.top, triangle.btm, pixels, index);
			CallRasterizeLine(triangle.mid, triangle.btm, pixels, index);
		}
	}
};
