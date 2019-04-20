/**
 * File Name : raster.cu
 * Author : Yang Fan
 * Date : 2018/11/28
 * declare raster
 */

#pragma once

#include <cmath>
#include <vector>
#include "../common/define.h"
#include <device_launch_parameters.h>
#include "../common/device.h"


/*
 * Bresenham Line Algorithm
 */
extern "C" void CallRasterizeLine(const Pixel& p1, const Pixel& p2, std::vector<Pixel>& pixels, size_t& index) {
	int maxSize = pixels.size();
	auto start = p1, end = p2;

	if (p1.pos._x > p2.pos._x) {
		start = p2;
		end = p1;
	}

	auto gad = 0.0f;
	auto disx = abs(end.pos._x - start.pos._x);
	auto disy = abs(end.pos._y - start.pos._y);
	auto dis = disx > disy ? disx : disy;

	Pixel p;
	for (auto i = 0; i <= dis; i++) {
		gad = i / dis;
		INTERPOLATEP(p1 , p2 , gad , p);
		if (index >= maxSize) {
			pixels.resize(maxSize * 1.5);
		}
		pixels[index++] = p;
	}
}


__global__ void RasterizeLines(Line* lines, int* baseIdx, Pixel* pixels , const int numPixels , const int numThreads) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numThreads) {

		const auto sp = lines[idx].left;
		const auto ep = lines[idx].right;

		const auto sx = sp.pos._x;
		const auto ex = ep.pos._x;

		for (auto x = sx; x <= ex; ++x) {
			float gad;
			if (ex - sx < 1.0f) { gad = 0; }
			else gad = (x - sx) / (ex - sx);

			const int index = baseIdx[idx] + (x - sx);
			if (index < numPixels) {
				INTERPOLATEP(sp , ep , gad , pixels[index]);
			}
		}
	}

}

extern "C" void CallRasterizeLines(const std::vector<Line>& lines, std::vector<Pixel>& pixels) {
	const int numLines = lines.size();
	const int numPixels = pixels.size();

	std::vector<int> baseIndexes(numLines , 0);

	for (auto i = 1; i < numLines; ++i) {
		baseIndexes[i] = baseIndexes[i - 1] + lines[i - 1].numPixels;
	}

	Line* dLines;
	CUDA_CALL(cudaMalloc(&dLines , sizeof(Line) * numLines));
	CUDA_CALL(cudaMemset(dLines , 0 , sizeof(Line) * numLines));

	int* dBaseIdx;
	CUDA_CALL(cudaMalloc(&dBaseIdx , sizeof(int) * numLines));
	CUDA_CALL(cudaMemset(dBaseIdx , 0 , sizeof(int) * numLines));

	Pixel* dPixels;
	CUDA_CALL(cudaMalloc(&dPixels , sizeof(Pixel) * numPixels));
	CUDA_CALL(cudaMemset(dPixels , 0 , sizeof(Pixel) * numPixels));

	CUDA_CALL(cudaMemcpy(dBaseIdx , &baseIndexes[0] , sizeof(int) * numLines , cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dLines , &lines[0] , sizeof(Line) * numLines , cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dPixels , &pixels[0] , sizeof(Pixel) * numPixels , cudaMemcpyHostToDevice));

	RasterizeLines<<<(numLines + 15) / 16, 16>>>(dLines , dBaseIdx , dPixels , numPixels , numLines);

	CUDA_CALL(cudaMemcpy(&pixels[0] , dPixels , sizeof(Pixel) * numPixels , cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(dLines));
	CUDA_CALL(cudaFree(dBaseIdx));
	CUDA_CALL(cudaFree(dPixels));

	baseIndexes.clear();
	baseIndexes.shrink_to_fit();
}