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

		shader.callVertexShader(vertex1 , pixel1);
		shader.callVertexShader(vertex2 , pixel2);
		shader.callVertexShader(vertex3 , pixel3);

		std::vector<Shader::PSInput> pixels;
		Raster::rasterize(pixel1 , pixel2 , pixel3 , pixels , type);
		std::vector<Color> colors(pixels.size());
		shader.callPixelShader(pixels , colors);

		for (auto i = 0 ; i < pixels.size(); ++i)
		{
			auto pixel = pixels[i];
			auto pixelColor = colors[i];
			if (!Device::getInstance().testDepth(pixel.pos._x, pixel.pos._y, pixel.pos._z)) continue; // Éî¶È²âÊÔ
			Device::getInstance().setPixel(pixel.pos._x, pixel.pos._y, pixelColor);
		}

		pixels.clear();
		pixels.shrink_to_fit();

		colors.clear();
		colors.shrink_to_fit();
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