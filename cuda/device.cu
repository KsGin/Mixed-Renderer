/**
 * File Name : device.cu
 * Author : Yang Fan
 * Date : 2019/4/2
 * define device mixed
 */

#pragma once
#include <device_launch_parameters.h>
#include "../common/device.h"
#include "../common/define.h"
#include "../common/texture.h"
#include <vector>

__device__ void intersect(const Ray& ray, const Triangle& triangle, IntersectResult& intersectResult) {
	const auto origin = ray.origin;
	const auto direction = ray.direction;

	const auto edge1 = triangle.top.pos3D - triangle.btm.pos3D;
	const auto edge2 = triangle.mid.pos3D - triangle.btm.pos3D;
	const auto normal = Math::Vector3::cross(edge1, edge2).normalize();

	const auto nd = Math::Vector3::dot(normal, direction);
	
	if (fabs(nd) < 0) return;

	const auto orp = origin - triangle.btm.pos3D;

	const auto no = - Math::Vector3::dot(normal, orp);

	const auto distance = no / nd;
	if (distance <= 0.0f) return;

	intersectResult.intersectPoint = origin + direction * distance;
	intersectResult.distance = distance;

	const auto edge11 = Math::Vector3::dot(edge1, edge1);
	const auto edge12 = Math::Vector3::dot(edge1, edge2);
	const auto edge22 = Math::Vector3::dot(edge2, edge2);

	const auto ot = intersectResult.intersectPoint - triangle.btm.pos3D;

	const auto ote1 = Math::Vector3::dot(ot, edge1);
	const auto ote2 = Math::Vector3::dot(ot, edge2);

	const auto d = edge12 * edge12 - edge11 * edge22;

	const auto s = (edge12 * ote2 - edge22 * ote1) / d;
	if (s < 0.0f || s > 1.0f) return;

	const auto t = (edge12 * ote1 - edge11 * ote2) / d;
	if (t < 0.0f || (s + t) > 1.0f) return;


	intersectResult.isSucceed = true;
}

/*
 * 设置颜色
 */
__device__ void SetPixel(int x, int y, const Color& color, Uint8* pixelColors, int screenWidth, int screenHeight) {
	auto r = color.r;
	auto g = color.g;
	auto b = color.b;
	auto a = color.a;
	CLAMP01(a);
	CLAMP01(b);
	CLAMP01(g);
	CLAMP01(r);

	auto i = (y * screenWidth + x) * 4;
	const auto size = (screenWidth - 1) * (screenHeight - 1) * 4;

	CLAMP(i , 4 , size);

	pixelColors[i - 1] = static_cast<Uint8>(a * 255);
	pixelColors[i - 2] = static_cast<Uint8>(b * 255);
	pixelColors[i - 3] = static_cast<Uint8>(g * 255);
	pixelColors[i - 4] = static_cast<Uint8>(r * 255);
}

/*
 * 设置颜色
 */
__device__ void GetPixel(int x, int y, Uint8* pixelColors, int screenWidth, int screenHeight, Color& color) {
	auto i = (y * screenWidth + x) * 4;
	const auto size = (screenWidth - 1) * (screenHeight - 1) * 4;

	CLAMP(i , 4 , size);

	color.a = pixelColors[i - 1] / static_cast<float>(255);
	color.b = pixelColors[i - 2] / static_cast<float>(255);
	color.g = pixelColors[i - 3] / static_cast<float>(255);
	color.r = pixelColors[i - 4] / static_cast<float>(255);
}

/*
 * 深度测试
 */
__device__ void TestDepth(int x, int y, float depth, float* depths, bool& isSuccess, int screenWidth,
                          int screenHeight) {
	if (x >= screenWidth || x <= 0 || y <= 0 || y >= screenHeight) return;

	const auto idx = y * screenWidth + x;
	const auto cdp = depths[idx];

	if (cdp == 0 || depth <= cdp) {
		depths[idx] = depth;
		isSuccess = true;
	}
}

/*
 * 计算光照
 */
__device__ void SampleLight(Pixel& pixel, Triangle* triangles, Color& color, int numTriangles) {
	if (pixel.sType == LIGHT) return;

	const auto lightOrigin = Math::Vector3(0, 2, 0);

	const auto lightDirection = (lightOrigin - pixel.pos3D).normalize();

	const auto distance = (lightOrigin - pixel.pos3D).length();

	const Math::Vector3 normal = pixel.normal.normalize();

	// 处理阴影
	Ray ray;
	ray.isActive = true;
	ray.origin = pixel.pos3D;
	ray.direction = lightDirection;

	bool isShadow = false;

#pragma unroll
	for (auto i = 0; i < numTriangles; ++i) {
		IntersectResult iTmp{false};
		intersect(ray, triangles[i], iTmp);
		if (iTmp.isSucceed && iTmp.distance > 0.05f && iTmp.distance < distance - 0.05) {
			isShadow = true;
			break;
		}
	}

	// 处理光照	
	float ambient = 0.3;

	float nd = Math::Vector3::dot(lightDirection, normal);
	CLAMP01(nd);

	color = color * (ambient + nd * !isShadow);
}

/*
 * 计算反射
 */
__device__ void SampleReflect(Pixel& pixel, Triangle* triangles, Color& color, int numTriangles) {
	Ray ray;
	ray.isActive = true;
	ray.origin = pixel.pos3D;
	ray.direction = pixel.normal.normalize();

	float minDistance = INT_MAX;

#pragma unroll
	for (auto i = 0; i < numTriangles; ++i) {
		IntersectResult iTmp{false};
		intersect(ray, triangles[i], iTmp);
		if (iTmp.isSucceed && iTmp.distance < minDistance) {
			minDistance = iTmp.distance;

			if (triangles[i].mid.sType == LIGHT) color = Color::white();
			else color = triangles[i].mid.color;
		}
	}
}


/*
 * 渲染管线混合阶段
 */
__global__ void KernelMixedReflect(Pixel* pixels, Color* colors, Triangle* triangles, Uint8* pixelColors, float* depths,
                                   int screenWidth, int screenHeight, int numTriangles, int numElements) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numElements) {

		const int x = pixels[idx].pos._x;
		const int y = pixels[idx].pos._y;

		auto isFirst = false;
		TestDepth(x, y, pixels[idx].pos._z, depths, isFirst, screenWidth, screenHeight);
		if (isFirst) {

			/*计算光照*/
			auto lightColor = Color::white();
			SampleLight(pixels[idx], triangles, lightColor, numTriangles);

			/*计算反射*/
			auto reflectColor = Color::black();
			SampleReflect(pixels[idx], triangles, reflectColor, numTriangles);

			/*混合颜色*/
			colors[idx] = (colors[idx] * (1 - pixels[idx].reflectiveness) + reflectColor * pixels[idx].reflectiveness) *
				lightColor;

			/*着色*/
			SetPixel(x, y, colors[idx], pixelColors, screenWidth, screenHeight);
		}
	}
}


/*
 * 渲染管线混合阶段
 */
__global__ void KernelMixed(Pixel* pixels, Color* colors, Uint8* pixelColors, float* depths,
                            int screenWidth, int screenHeight, int numElements) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numElements) {

		const int x = pixels[idx].pos._x;
		const int y = pixels[idx].pos._y;

		auto isFirst = false;
		TestDepth(x, y, pixels[idx].pos._z, depths, isFirst, screenWidth, screenHeight);
		if (isFirst) {
			/*着色*/
			SetPixel(x, y, colors[idx], pixelColors, screenWidth, screenHeight);
		}
	}
}


extern "C" void CallMixed(std::vector<Pixel>& pixels, std::vector<Color>& colors,
                          const std::vector<Triangle>& triangles, Uint8* pixelColors, float* depths, int screenWidth,
                          int screenHeight) {
	if (pixels.empty()) return;

	const int numPixels = pixels.size();
	const int numTriangles = triangles.size();
	const int screenPixelSize = screenWidth * screenHeight;

	Pixel* dPixels;
	CUDA_CALL(cudaMalloc(&dPixels , sizeof(Pixel) * numPixels));
	CUDA_CALL(cudaMemset(dPixels , 0 , sizeof(Pixel) * numPixels));
	Color* dColors;
	CUDA_CALL(cudaMalloc(&dColors , sizeof(Color) * numPixels));
	CUDA_CALL(cudaMemset(dColors , 0 , sizeof(Color) * numPixels));
	Uint8* dPixelColors;
	CUDA_CALL(cudaMalloc(&dPixelColors , sizeof(Uint8) * screenPixelSize * 4));
	CUDA_CALL(cudaMemset(dPixelColors , 0 , sizeof(Uint8) * screenPixelSize * 4));
	float* dDepths;
	CUDA_CALL(cudaMalloc(&dDepths , sizeof(float) * screenPixelSize));
	CUDA_CALL(cudaMemset(dDepths , 0 , sizeof(float) * screenPixelSize));

	Triangle* dTriangles;
	CUDA_CALL(cudaMalloc(&dTriangles , sizeof(Triangle) * numTriangles));
	CUDA_CALL(cudaMemset(dTriangles , 0 , sizeof(Triangle) * numTriangles));


	CUDA_CALL(cudaMemcpy(dPixelColors , pixelColors , sizeof(Uint8) * screenPixelSize * 4 , cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dDepths , depths , sizeof(float) * screenPixelSize , cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dPixels , &pixels[0] , sizeof(Pixel) * numPixels , cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dColors , &colors[0] , sizeof(Color) * numPixels , cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dTriangles , &triangles[0] , sizeof(Triangle) * numTriangles , cudaMemcpyHostToDevice));

	// 着色
	KernelMixedReflect<<<(numPixels + 255) / 256 , 256>>>(dPixels, dColors, dTriangles, dPixelColors, dDepths,
	                                                      screenWidth,
	                                                      screenHeight, numTriangles, numPixels);

	cudaDeviceSynchronize();

	CUDA_CALL(cudaMemcpy(depths , dDepths , sizeof(float) * screenPixelSize , cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(pixelColors , dPixelColors , sizeof(Uint8) * screenPixelSize * 4, cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(dDepths));
	CUDA_CALL(cudaFree(dPixels));
	CUDA_CALL(cudaFree(dColors));
	CUDA_CALL(cudaFree(dPixelColors));
}

