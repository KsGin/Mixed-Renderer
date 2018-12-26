/**
 * File Name : render.h
 * Author : Yang Fan
 * Date : 2018/12/20
 * render pipe
 */

#pragma once
#include "model.h"
#include "shader.h"
#include "raster.h"
#include "device.h"

class Render
{
	static void doRenderFace(const Model::Mesh::Face& face, const Shader &shader , const Raster::TYPE &type) {
		Shader::VSInput vertex1 , vertex2 , vertex3;

		vertex1.pos = face.v1.pos;
		vertex2.pos = face.v2.pos;
		vertex3.pos = face.v3.pos;

		vertex1.normal = face.v1.normal;
		vertex2.normal = face.v2.normal;
		vertex3.normal = face.v3.normal;

		vertex1.color = face.v1.color;
		vertex2.color = face.v2.color;
		vertex3.color = face.v3.color;

		Shader::PSInput pixel1 , pixel2 , pixel3;

		pixel1 = shader.vertexShader(vertex1);
		pixel2 = shader.vertexShader(vertex2);
		pixel3 = shader.vertexShader(vertex3);

		std::vector<Shader::PSInput> pixels;
		Raster::rasterize(pixel1 , pixel2 , pixel3 , pixels , type);

		for (auto& pixel : pixels)
		{
			Color pixelColor = shader.pixelShader(pixel);
			if (!Device::getInstance().testDepth(pixel.pos._x, pixel.pos._y, pixel.pos._z)) continue; // …Ó∂»≤‚ ‘
			Device::getInstance().setPixel(pixel.pos._x, pixel.pos._y, pixelColor);
		}

		pixels.clear();
		pixels.shrink_to_fit();
	}

public:
	static void render(const Model &model , const Shader &shader, const Raster::TYPE &type = Raster::TYPE::SOLID) {
		for (auto& mesh : model.meshes) {
			for (auto& face : mesh.faces){
				doRenderFace(face , shader , type);
			}
		}
	}
};