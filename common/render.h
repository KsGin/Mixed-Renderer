/**
 * File Name : render.h
 * Author : Yang Fan
 * Date : 2018/12/20
 * render pipe
 */

#pragma once

#include "../common/define.h"
#include "device.h"
#include "model.h"
#include "shader.h"
#include "raster.h"
#include "camera.h"

class Render {
	static void doRenderVertex(const Model& model,const Shader& shader) {
		auto idx = 0;
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

				if (idx >= triangles.size()) {
					triangles.resize(idx * 1.5);
				}
				triangles[idx++] = triangle;
			}
		}
	}

	static void doRasterize(const TYPE& type = SOLID ) {
		//GPU
		Raster::doRasterize(triangles , pixels , type);

	}

	static void doRenderPixel(const Shader& shader) {
		//GPU
		shader.pixelShader(pixels, colors);

	}

	static void doMixed() {
		//GPU
		Device::getInstance().mixed(pixels, colors);
	}

	/*
	 * 为效率妥协使用 memset
	 */
	static void doReset() {
		memset(&pixels[0] , 0 , sizeof(Pixel) * pixels.size());
		memset(&colors[0] , 0 , sizeof(Color) * colors.size());
		memset(&triangles[0] , 0 , sizeof(Triangle) * triangles.size());
	}

public:
	
	static std::vector<Pixel> pixels;

	static std::vector<Color> colors;

	static std::vector<Triangle> triangles;

	static bool isInit;

	static void initialize() {
		pixels.resize(25600);
		colors.resize(25600);
		triangles.resize(256);
	}

	static void render(const Model& model , const Camera& camera , Shader& shader , const TYPE& type = SOLID) {

		if(!isInit) {
			initialize();
			isInit = true;
		}
				
		doReset();

		doRenderVertex(model , shader);

		doRasterize(type);

		doRenderPixel(shader);

		doMixed();
	}
};

std::vector<Pixel> Render::pixels;
std::vector<Color> Render::colors;
std::vector<Triangle> Render::triangles;
bool Render::isInit = false;
