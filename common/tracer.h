/**
 * File Name : tracer.h
 * Author : Yang Fan
 * Date : 2019/5/5
 * declare tracing class
 */

#pragma once
#include "define.h"

extern "C" void CallTracing(const std::vector<Ray>& rays, const std::vector<Triangle>& triangles, std::vector<IntersectResult>& intersectResults);

class Tracer {
	

public:

	void tracing(const std::vector<Ray>& rays , const std::vector<Triangle>& triangles , std::vector<IntersectResult>& intersectResults) {
		CallTracing(rays , triangles , intersectResults);
	}

	static Tracer& getInstance() {
		static Tracer tracer;
		return tracer;
	}
};
