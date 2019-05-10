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
		//GPU
		Device::getInstance().mixed(pixels, colors);
	}

	void generateRay() {
		
	}

	void tracing() {
		Tracer::getInstance().tracing(getPerspectiveCamera() , preTriangles, intersectResults);
	}

	/*
	 * ΪЧ����Эʹ�� memset
	 */
	void reset() {

		triangles.clear();

		memset(&pixels[0], 0, sizeof(Pixel) * pixels.size());
		memset(&colors[0], 0, sizeof(Color) * colors.size());
	}


	Renderer() {

	}

public:

	/*
	 * ����
	 */
	static Renderer& getInstance() {
		static Renderer instance;
		return instance;
	}

	/*
	 * ��ʼ��
	 */
	static void initialize(const int& numPixels, const int& numColors, const int& numTriangles, const int& numModels) {

		auto& d = Device::getInstance();

		auto& r = getInstance();
		r.pixels = std::vector<Pixel>(numPixels);
		r.colors = std::vector<Color>(numColors);
		r.triangles = std::vector<Triangle>(numTriangles);
		r.preTriangles = std::vector<Triangle>(numTriangles);
		r.intersectResults = std::vector<IntersectResult>(d.width * d.height);

		r.pixels.resize(numPixels);
		r.colors.resize(numColors);
		r.intersectResults.resize(d.width * d.height);

		r.renderModels = std::vector<RenderModel>(0);
		r.renderModels.reserve(numModels);
	}

	/*
	 * ���������
	 */
	void setPerspectiveCamera(const PerspectiveCamera& camera) {
		mainPerspectiveCamera = camera;
	}

	/*
	 * ��������
	 */
	PerspectiveCamera getPerspectiveCamera() const {
		return mainPerspectiveCamera;
	}


	/*
	 * ��ģ�������������
	 */
	void add(const Model& model, const Shader& shader, const RenderType& type = SOLID) {
		renderModels.emplace_back(RenderModel{model, shader, type});
	}

	void clear() {
		renderModels.clear();
		preTriangles.clear();

		memset(&intersectResults[0], 0, sizeof(IntersectResult) * intersectResults.size());
	}

	/*
	 * ������Ⱦ
	 */
	void render() {
		// ��դ����
		for (auto& m : renderModels) {
			reset();

			renderVertex(m.model, m.shader);

			rasterize(m.type);

			renderPixel(m.shader);

			mixed();
		}

		// ����׷�ٹ���
		tracing();
	}

	/*
	 * ����
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

		preTriangles.clear();
		preTriangles.shrink_to_fit();
	}
};
