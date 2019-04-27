/**
 * File Name : doRenderPipe.h
 * Author : Yang Fan
 * Date : 2018/12/20
 * doRenderPipe pipe
 */

#pragma once

#include "../common/define.h"
#include "device.h"
#include "model.h"
#include "shader.h"
#include "raster.h"
#include "camera.h"

class Renderer {

	std::vector<Pixel> pixels;
	
	std::vector<Color> colors;
	
	std::vector<Triangle> triangles;

	void doRenderVertex(const Model& model,const Shader& shader) {
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

	void doRasterize(const TYPE& type = SOLID ) {
		//GPU
		Raster::doRasterize(triangles , pixels , type);

	}

	void doRenderPixel(const Shader& shader) {
		//GPU
		shader.pixelShader(pixels, colors);

	}

	void doMixed() {
		//GPU
		Device::getInstance().mixed(pixels, colors);
	}

public:

	/*
	 * 单例
	 */
	static Renderer& getInstance() {
		static Renderer instance;
		return instance;
	}

	static void initialize(const int & numPixels , const int & numColors , const int & numTriangles) {
		auto& r = getInstance();
		r.pixels = std::vector<Pixel>(numPixels);
		r.colors = std::vector<Color>(numColors);
		r.triangles = std::vector<Triangle>(numTriangles);

		r.pixels.resize(numPixels);
		r.colors.resize(numColors);
		r.triangles.resize(numTriangles);
	}

	/*
	 * 为效率妥协使用 memset
	 */
	void doResetPipe() {
		memset(&pixels[0] , 0 , sizeof(Pixel) * pixels.size());
		memset(&colors[0] , 0 , sizeof(Color) * colors.size());
		memset(&triangles[0] , 0 , sizeof(Triangle) * triangles.size());
	}

	void doAddPipe(const Model& model , const Camera& camera , Shader& shader , const TYPE& type = SOLID) {

		doResetPipe();

		doRenderVertex(model , shader);

		doRasterize(type);

		doRenderPixel(shader);

		doMixed();
	}

	void doRenderPipe() {

	
	}

	void destory() {
		pixels.clear();
		pixels.shrink_to_fit();

		colors.clear();
		colors.shrink_to_fit();

		triangles.clear();
		triangles.shrink_to_fit();
	}
};
