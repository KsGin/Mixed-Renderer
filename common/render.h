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

	std::vector<Pixel> pixels;

	std::vector<Color> colors;

	std::vector<Triangle> triangles;

	std::vector<IntersectResult> intersectResults;

	std::unordered_map<ShaderType , ShaderData> shaderMap;

	PerspectiveCamera mainPerspectiveCamera;

private:

	void renderVertex(const Model& model, const Shader& shader) {
		for (auto& mesh : model.meshes) {
			for (auto& face : mesh.faces) {
				Triangle triangle;

				vertexShader(shaderMap , face.v1, triangle.top , shader.sData.sType);
				vertexShader(shaderMap , face.v2, triangle.mid , shader.sData.sType);
				vertexShader(shaderMap , face.v3, triangle.btm , shader.sData.sType);

				triangle.top.sType = triangle.mid.sType = triangle.btm.sType = shader.sData.sType;
				triangles.emplace_back(triangle);
			}
		}
	}

	void rasterize(const RenderType& type = SOLID) {
		//GPU
		Raster::getInstance().doRasterize(triangles, pixels, type);

	}

	void renderPixel() {
		//GPU
		pixelShader(shaderMap , pixels, colors);
	}

	void mixed() {
		//GPU
		Device::getInstance().mixed(pixels, colors);
	}

	void generateRay() {

	}

	void tracing() {
		Tracer::getInstance().tracing(getPerspectiveCamera(), triangles, intersectResults);
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
	static void initialize(const int& numPixels, const int& numColors, const int& numTriangles) {

		auto& d = Device::getInstance();

		auto& r = getInstance();
		r.pixels = std::vector<Pixel>(numPixels);
		r.colors = std::vector<Color>(numColors);
		r.triangles = std::vector<Triangle>(numTriangles);
		r.intersectResults = std::vector<IntersectResult>(d.width * d.height);

		r.pixels.resize(numPixels);
		r.colors.resize(numColors);
		r.intersectResults.resize(d.width * d.height);
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
 * 为效率妥协使用 memset
 */
	void clear() {

		triangles.clear();
		shaderMap.clear();

		memset(&intersectResults[0], 0, sizeof(IntersectResult) * intersectResults.size());
		memset(&pixels[0], 0, sizeof(Pixel) * pixels.size());
		memset(&colors[0], 0, sizeof(Color) * colors.size());
	}

	/*
	 * 将模型添加至管线中
	 */
	void add(const Model& model, const Shader& shader, const RenderType& type = SOLID) {	
		shaderMap[shader.sData.sType] = shader.sData;
		renderVertex(model , shader);
	}

	/*
	 * 管线渲染
	 */
	void render() {

		rasterize();

		renderPixel();

		mixed();

		// // 光线追踪过程
		// tracing();
	}

	/*
	 * 销毁
	 */
	void destory() {
		intersectResults.clear();
		intersectResults.shrink_to_fit();

		pixels.clear();
		pixels.shrink_to_fit();

		colors.clear();
		colors.shrink_to_fit();

		triangles.clear();
		triangles.shrink_to_fit();
	}
};
