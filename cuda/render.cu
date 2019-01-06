/**
 * File Name : render.h
 * Author : Yang Fan
 * Date : 2018/12/20
 * render pipe
 */

#pragma once
#include "../cuda/shader.cu"
#include "../cuda/raster.cu"
#include "../common/device.h"
#include "../common/model.h"

class Render
{
	static void doRenderFace(const Model::Mesh::Face& face, Shader &shader , const Raster::TYPE &type) {
		Shader::VSInput vertex1 , vertex2 , vertex3;

		vertex1.pos = face.v1.pos;
		vertex2.pos = face.v2.pos;
		vertex3.pos = face.v3.pos;

		vertex1.normal = face.v1.normal;
		vertex2.normal = face.v2.normal;
		vertex3.normal = face.v3.normal;

		vertex1.uv = face.v1.uv;
		vertex2.uv = face.v2.uv;
		vertex3.uv = face.v3.uv;

		vertex1.color = face.v1.color;
		vertex2.color = face.v2.color;
		vertex3.color = face.v3.color;

		Shader::PSInput pixel1 , pixel2 , pixel3;

		shader.vertexShader(vertex1 , pixel1);
		shader.vertexShader(vertex2 , pixel2);
		shader.vertexShader(vertex3 , pixel3);

		std::vector<Shader::PSInput> hPixels;
		Raster::rasterize(pixel1 , pixel2 , pixel3 , hPixels , type);
		Shader::PSInput *dPixels;
		CUDA_CALL(cudaMalloc((void**)&dPixels , sizeof(Shader::PSInput) * hPixels.size()));

		Color* hPixelsColor = static_cast<Color*>(malloc(sizeof(Color) * hPixels.size()));
		Color* dPixelsColor;
		CUDA_CALL(cudaMalloc((void**)&dPixelsColor , sizeof(Color) * hPixels.size()));
		CUDA_CALL(cudaMemcpy(dPixels , &hPixels[0] , hPixels.size() , cudaMemcpyHostToDevice));

		shader.pixelShader<<<1 , hPixels.size()>>>(&dPixels[0] , dPixelsColor);

		CUDA_CALL(cudaMemcpy(hPixelsColor , dPixelsColor , hPixels.size() , cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaFree(dPixels));
		CUDA_CALL(cudaFree(dPixelsColor));

		for (auto i = 0 ; i < hPixels.size(); ++i)
		{
			auto pixel = hPixels[i];
			auto pixelColor = hPixelsColor[i];
			if (!Device::getInstance().testDepth(pixel.pos._x, pixel.pos._y, pixel.pos._z)) continue; // Éî¶È²âÊÔ
			Device::getInstance().setPixel(pixel.pos._x, pixel.pos._y, pixelColor);
		}

		free(hPixelsColor);

		hPixels.clear();
		hPixels.shrink_to_fit();
	}

public:
	static void render(const Model &model , Shader &shader, const Raster::TYPE &type = Raster::TYPE::SOLID) {
		for (auto& mesh : model.meshes) {
			for (auto& face : mesh.faces){
				doRenderFace(face , shader , type);
			}
		}
	}
};