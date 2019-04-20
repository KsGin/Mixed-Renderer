/**
 * File Name : render.h
 * Author : Yang Fan
 * Date : 2018/12/20
 * render pipe
 */

#pragma once

#include "../common/define.h"
#include "../includes/math/vector.hpp"
#include "device.h"
#include "model.h"
#include "shader.h"
#include "raster.h"

class Render {
	static void doRenderVertex(const Model& model, Shader& shader, std::vector<Triangle>& triangles) {
		for (auto& mesh : model.meshes) {
			for (auto& face : mesh.faces) {
				Triangle triangle;

				Vertex vertex1, vertex2, vertex3;
				vertex1 = face.v1;
				vertex2 = face.v2;
				vertex3 = face.v3;

				shader.vertexShader(vertex1, triangle.top);
				shader.vertexShader(vertex2, triangle.mid);
				shader.vertexShader(vertex3, triangle.btm);

				triangles.emplace_back(triangle);
			}
		}
	}

	static void doRasterize(std::vector<Triangle>& triangles, std::vector<Pixel>& pixels,const TYPE& type = SOLID ) {
		Raster::doRasterize(triangles, pixels);
	}

	static void doRenderPixel(Shader& shader, std::vector<Pixel>& pixels) {
		std::vector<Color> colors(pixels.size());

		//GPU
		shader.pixelShader(pixels, colors);

		//GPU
		Device::getInstance().mixed(pixels, colors);

		colors.clear();
		colors.shrink_to_fit();
	}
public:
	static void render(const Model& model, Shader& shader, const TYPE& type = TYPE::SOLID) {
		std::vector<Triangle> triangles;
		doRenderVertex(model , shader , triangles);

		std::vector<Pixel> pixels;
		doRasterize(triangles, pixels);

		doRenderPixel(shader, pixels);

		pixels.clear();
		pixels.shrink_to_fit();

		triangles.clear();
		triangles.shrink_to_fit();
	}
};
