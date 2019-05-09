/**
 * File Name : tracer.cu
 * Author : Yang Fan
 * Date : 2019/5/8
 * define tracer
 */


#include "../common/define.h"
#include <vector>
#include "../common/ray.h"
#include <device_launch_parameters.h>
#include "../common/camera.h"

__device__ void intersect(const Ray& ray, const Triangle& triangle, IntersectResult& intersectResult) {

	const auto origin = ray.origin;
	const auto direction = ray.direction;

	const auto edge1 = triangle.top.pos - triangle.btm.pos;
	const auto edge2 = triangle.mid.pos - triangle.btm.pos;
	const auto normal = Math::Vector3::cross(edge1, edge2).normalize();

	const auto b = Math::Vector3::dot(normal, direction);

	if (fabs(b) < 0) return;

	const auto w0 = origin - triangle.btm.pos;

	const auto a = -Math::Vector3::dot(normal, w0);

	const auto r = a / b;
	if (r < 0.0f) return;

	intersectResult.intersectPoint = origin + direction * r;
	intersectResult.distance = r;

	const auto e11 = Math::Vector3::dot(edge1, edge1);
	const auto e12 = Math::Vector3::dot(edge1, edge2);
	const auto e22 = Math::Vector3::dot(edge2, edge2);

	const auto w = intersectResult.intersectPoint - triangle.btm.pos;

	const auto w1 = Math::Vector3::dot(w, edge1);
	const auto w2 = Math::Vector3::dot(w, edge2);

	const auto d = e12 * e12 - e11 * e22;

	const auto s = (e12 * w2 - e22 * w1) / d;
	if (s < 0.0f || s > 1.0f) return;

	const auto t = (e12 * w1 - e11 * w2) / d;
	if (t < 0.0f || (s + t) > 1.0f) return;


	intersectResult.isSucceed = true;
}

__global__ void KernelTracing(Ray* rays , Triangle* triangles , IntersectResult* intersectResults , int numRays , int numTriangles , int numIntersectResults) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numRays) {
		if (!rays[idx].isActive) return;
		float minDistance = INT_MAX;
		IntersectResult tmpIntersectResult {false};
		for (auto i = 0 ; i < numTriangles; ++i) {
			intersect(rays[idx] , triangles[i] , tmpIntersectResult);
			if (tmpIntersectResult.isSucceed && tmpIntersectResult.distance < minDistance) {
				intersectResults[idx] = tmpIntersectResult;
				minDistance = tmpIntersectResult.distance;
			}
		}
	}
}

extern "C" void CallTracing(const PerspectiveCamera& perspectiveCamera , const std::vector<Triangle>& triangles, std::vector<IntersectResult>& intersectResults) {
	if (triangles.empty() || intersectResults.empty()) return;

	const auto numTriangles = triangles.size();
	const auto numIntersectResults = intersectResults.size();

	Triangle* dTriangles;
	CUDA_CALL(cudaMalloc(&dTriangles , sizeof(Triangle) * numTriangles));
	CUDA_CALL(cudaMemset(dTriangles , 0 , sizeof(Triangle) * numTriangles));
	IntersectResult* dIntersectResults;
	CUDA_CALL(cudaMalloc(&dIntersectResults , sizeof(IntersectResult) * numIntersectResults));
	CUDA_CALL(cudaMemset(dIntersectResults , 0 , sizeof(IntersectResult) * numIntersectResults));

	CUDA_CALL(cudaMemcpy(dTriangles , &triangles[0] , sizeof(Triangle) * numTriangles , cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dIntersectResults , &intersectResults[0] , sizeof(IntersectResult) * numIntersectResults , cudaMemcpyHostToDevice));

	// KernelTracing<<<(numRays + 63) / 64 , 64>>>(dTriangles , dIntersectResults , numTriangles , numIntersectResults);

	CUDA_CALL(cudaMemcpy(&intersectResults[0] , dIntersectResults , sizeof(IntersectResult) * numIntersectResults , cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(dTriangles));
	CUDA_CALL(cudaFree(dIntersectResults));
}
