/**
 * File Name : render.h
 * Author : Yang Fan
 * Date : 2018/12/20
 * render pipe
 */

#pragma once
#include "../common/device.h"
#include "../common/model.h"
#include "../common/color.h"
#include "../common/define.h"
#include "../common/shader.h"
#include "../cuda/raster.cu"

class Render
{
	static void doRasterizeFace(const Model::Mesh::Face& face, Shader &shader , const TYPE &type , std::vector<Pixel>& pixels) {
		Vertex vertex1 , vertex2 , vertex3;

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

		Pixel pixel1 , pixel2 , pixel3;

		shader.vertexShader(vertex1 , pixel1);
		shader.vertexShader(vertex2 , pixel2);
		shader.vertexShader(vertex3 , pixel3);

		//CPU
		Raster::rasterize(pixel1 , pixel2 , pixel3 , pixels , type);
	}

	static void doRenderPixel(Shader &shader , std::vector<Pixel> &pixels) {
		std::vector<Color> colors(pixels.size());

		//GPU
		shader.pixelShader(pixels , colors);

		//GPU
		Device::getInstance().mixed(pixels , colors);

		colors.clear();
		colors.shrink_to_fit();
	}

public:
	static void render(const Model &model , Shader &shader, const TYPE &type = TYPE::SOLID) {
		std::vector<Pixel> pixels;
		for (auto& mesh : model.meshes) {
			for (auto& face : mesh.faces){
				doRasterizeFace(face , shader , type , pixels);
			}
		}
		doRenderPixel(shader , pixels);

		pixels.clear();
		pixels.shrink_to_fit();
	}
};
