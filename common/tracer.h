/**
 * File Name : tracer.h
 * Author : Yang Fan
 * Date : 2019/5/5
 * declare tracing class
 */

#pragma once
#include "define.h"

extern "C" void CallTracing(const PerspectiveCamera& perspectiveCamera, const std::vector<Triangle>& triangles,
                            std::vector<IntersectResult>& intersectResults);

class Tracer {

	void intersect(const Ray& ray, const Triangle& triangle, IntersectResult& intersectResult) {
		const auto origin = ray.origin;
		const auto direction = ray.direction;
		
		const auto u = triangle.top.pos3D - triangle.btm.pos3D;
		const auto v = triangle.mid.pos3D - triangle.btm.pos3D;
		const auto norm = Math::Vector3::cross(u, v).normalize();
		
		const auto b = Math::Vector3::dot(norm, direction);
		
		if (fabs(b) < 0) return;
		
		const auto w0 = origin - triangle.btm.pos3D;
		
		const auto a = -Math::Vector3::dot(norm, w0);
		
		const auto r = a / b;
		if (r < 0.0f) return;
		
		intersectResult.intersectPoint = origin + direction * r;
		intersectResult.distance = r;
		
		const auto uu = Math::Vector3::dot(u, u);
		const auto uv = Math::Vector3::dot(u, v);
		const auto vv = Math::Vector3::dot(v, v);
		
		const auto w = intersectResult.intersectPoint - triangle.btm.pos3D;
		
		const auto wu = Math::Vector3::dot(w, u);
		const auto wv = Math::Vector3::dot(w, v);
		
		const auto d = uv * uv - uu * vv;
		
		const auto s = (uv * wv - vv * wu) / d;
		if (s < 0.0f || s > 1.0f) return;
		
		const auto t = (uv * wu - uu * wv) / d;
		if (t < 0.0f || (s + t) > 1.0f) return;
		
		
		intersectResult.isSucceed = true;
	}


public:

	void tracing(const PerspectiveCamera& perspectiveCamera , const std::vector<Triangle>& triangles,
	             std::vector<IntersectResult>& intersectResults) {

		CallTracing(perspectiveCamera , triangles , intersectResults);
		//
		// const auto w = Device::getInstance().width;
		// const auto h = Device::getInstance().height;
		// 	
		// for (auto i = 0 ; i < w; ++i) {
		// 	auto dx = i / static_cast<float>(w);
		// 	for (auto j = 0 ; j < h; ++j) {
		// 		auto dy = 1 - j / static_cast<float>(h);
		// 		Ray ray;
		// 		perspectiveCamera.generateRay(dx , dy , ray);
		// 		float min = INT_MAX;
		// 		for (auto& triangle : triangles) {
		// 			IntersectResult intersectResult{false, 0};
		// 			intersect(ray, triangle, intersectResult);
		// 			if (intersectResult.isSucceed && intersectResult.distance < min) {
		// 				min = intersectResult.distance;
		// 				intersectResults[i * w + j] = intersectResult;
		// 			}
		// 		}
		// 	}
		// }
	}

	static Tracer& getInstance() {
		static Tracer tracer;
		return tracer;
	}
};
