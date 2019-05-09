/**
 * File Name : doRenderPipe.h
 * Author : Yang Fan
 * Date : 2018/12/20
 * doRenderPipe pipe
 */

#pragma once

#include "../common/define.h"
#include "device.h"
#include "raster.h"
#include "camera.h"
#include <unordered_map>
#include "tracer.h"
#include "model.h"
#include "shader.h"

class Renderer {

	struct RenderModel {
		Model model;
		Shader shader;
		RenderType type;
	};

	std::vector<Pixel> pixels;

	std::vector<Color> colors;

	std::vector<Triangle> triangles;

	std::vector<Triangle> preTriangles;

	std::vector<Ray> rays;

	std::vector<RenderModel> renderModels;

	std::vector<IntersectResult> intersectResults;

	PerspectiveCamera mainPerspectiveCamera;

private:
	void renderVertex(const Model& model, const Shader& shader) {
		for (auto& mesh : model.meshes) {
			for (auto& face : mesh.faces) {
				Triangle preTriangle;
				Triangle triangle;	

				preTriangle.top = face.v1;
				preTriangle.mid = face.v2;
				preTriangle.btm = face.v3;

				shader.vertexShader(preTriangle.top, triangle.top);
				shader.vertexShader(preTriangle.mid, triangle.mid);
				shader.vertexShader(preTriangle.btm, triangle.btm);

				preTriangle.top.pos3D = triangle.top.pos3D;
				preTriangle.mid.pos3D = triangle.mid.pos3D;
				preTriangle.btm.pos3D = triangle.btm.pos3D;

				preTriangles.emplace_back(preTriangle);
				triangles.emplace_back(triangle);
			}
		}
	}

	void rasterize(const RenderType& type = SOLID) {
		//GPU
		Raster::getInstance().doRasterize(triangles, pixels, type);

	}

	void renderPixel(const Shader& shader) {
		//GPU
		shader.pixelShader(pixels, colors);

	}

	void mixed() {

		if (rays.size() < pixels.size()) {
			rays.resize(pixels.size());
		}

		//GPU
		Device::getInstance().mixed(pixels, colors, rays, getPerspectiveCamera());
	}

	void tracing() {
		Tracer::getInstance().tracing(rays, triangles, intersectResults);
	}

	/*
	 * 为效率妥协使用 memset
	 */
	void reset() {

		triangles.clear();

		memset(&pixels[0], 0, sizeof(Pixel) * pixels.size());
		memset(&colors[0], 0, sizeof(Color) * colors.size());
		memset(&rays[0], 0, sizeof(Ray) * rays.size());
		memset(&intersectResults[0], 0, sizeof(IntersectResult) * intersectResults.size());
	}


	Renderer() {

	}

public:

	/*
	 * 单例
	 */
	static Renderer& getInstance() {
		static Renderer instance;
		return instance;
	}

	/*
	 * 初始化
	 */
	static void initialize(const int& numPixels, const int& numColors, const int& numTriangles, const int& numModels) {
		auto& r = getInstance();
		r.pixels = std::vector<Pixel>(numPixels);
		r.colors = std::vector<Color>(numColors);
		r.triangles = std::vector<Triangle>(numTriangles);
		r.preTriangles = std::vector<Triangle>(numTriangles);
		r.rays = std::vector<Ray>(numPixels);
		r.intersectResults = std::vector<IntersectResult>(numPixels);

		r.pixels.resize(numPixels);
		r.colors.resize(numColors);
		r.rays.resize(numPixels);
		r.rays.resize(numPixels);

		r.renderModels = std::vector<RenderModel>(0);
		r.renderModels.reserve(numModels);
	}

	/*
	 * 设置摄像机
	 */
	void setPerspectiveCamera(const PerspectiveCamera& camera) {
		mainPerspectiveCamera = camera;
	}

	/*
	 * 获得摄像机
	 */
	PerspectiveCamera getPerspectiveCamera() const {
		return mainPerspectiveCamera;
	}


	/*
	 * 将模型添加至管线中
	 */
	void add(const Model& model, const Shader& shader, const RenderType& type = SOLID) {
		renderModels.emplace_back(RenderModel{model, shader, type});
	}

	void clear() {
		renderModels.clear();
		preTriangles.clear();
	}

	/*
	 * 管线渲染
	 */
	void render() {
		// 光栅过程
		for (auto& m : renderModels) {
			reset();

			renderVertex(m.model, m.shader);

			rasterize(m.type);

			renderPixel(m.shader);

			mixed();
		}

		// 光线追踪过程
		tracing();
	}

	/*
	 * 销毁
	 */
	void destory() {
		intersectResults.clear();
		intersectResults.shrink_to_fit();

		rays.clear();
		rays.shrink_to_fit();

		pixels.clear();
		pixels.shrink_to_fit();

		colors.clear();
		colors.shrink_to_fit();

		triangles.clear();
		triangles.shrink_to_fit();

		preTriangles.clear();
		preTriangles.shrink_to_fit();
	}
};
