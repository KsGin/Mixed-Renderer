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
	if (lines.empty() || pixels.empty()) return;

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

	RasterizeLines<<<(numLines + 63) / 64, 64>>>(dLines , dBaseIdx , dPixels , numPixels , numLines);

	cudaDeviceSynchronize();

	CUDA_CALL(cudaMemcpy(&pixels[0] , dPixels , sizeof(Pixel) * numPixels , cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(dLines));
	CUDA_CALL(cudaFree(dBaseIdx));
	CUDA_CALL(cudaFree(dPixels));

	baseIndexes.clear();
	baseIndexes.shrink_to_fit();
}