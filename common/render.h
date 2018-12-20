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
		Shader::Vertex vertex1 , vertex2 , vertex3;

		vertex1.pos = face.p1;
		vertex2.pos = face.p2;
		vertex3.pos = face.p3;

		vertex1.normal = face.normal;
		vertex2.normal = face.normal;
		vertex3.normal = face.normal;

		vertex1.color = face.color;
		vertex2.color = face.color;
		vertex3.color = face.color;

		Shader::Pixel pixel1 , pixel2 , pixel3;

		pixel1 = shader.vertexShader(vertex1);
		pixel2 = shader.vertexShader(vertex2);
		pixel3 = shader.vertexShader(vertex3);

		std::vector<Shader::Pixel> pixels;
		Raster::raster(pixel1 , pixel2 , pixel3 , pixels , type);

		for (auto& pixel : pixels)
		{
			Color pixelColor = shader.pixelShader(pixel);
			Device::getInstance().setPixelColor(pixel.pos._x, pixel.pos._y, pixelColor);
		}

	}

public:
	static void render(const Model &model , const Shader &shader, const Raster::TYPE &type) {
		for (auto& mesh : model.meshes) {
			for (auto& face : mesh.faces){
				doRenderFace(face , shader , type);
			}
		}
	}
};