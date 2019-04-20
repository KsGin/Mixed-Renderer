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
	/*
	 * Distance of two points
	 */
	static int Distance(const Math::Vector3& p1, const Math::Vector3& p2) {
		return sqrt(pow(p2._x - p1._x, 2) + pow(p2._y - p1._y, 2));
	}

	/*
	 * Fixed point function
	 */
	static Math::Vector3 FixedPoint(const Math::Vector3& p) {
		Math::Vector3 refP;
		refP._x = floorf(-p._x * Device::getInstance().width + Device::getInstance().width / 2);
		refP._y = floorf(-p._y * Device::getInstance().height + Device::getInstance().height / 2);
		refP._z = p._z;
		return refP;
	}

	static void doRenderFace(const Model& model, Shader& shader, const TYPE& type,
	                         std::vector<Triangle>& triangles, int& numPixels) {

		for (auto& mesh : model.meshes) {
			for (auto& face : mesh.faces) {

				Triangle triangle;

				Vertex vertex1, vertex2, vertex3;

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

				Pixel pixel1, pixel2, pixel3;

				shader.vertexShader(vertex1, pixel1);
				shader.vertexShader(vertex2, pixel2);
				shader.vertexShader(vertex3, pixel3);

				auto top = pixel1;
				top.pos = FixedPoint(top.pos);
				auto mid = pixel2;
				mid.pos = FixedPoint(mid.pos);
				auto btm = pixel3;
				btm.pos = FixedPoint(btm.pos);

				Pixel tmp;
				// 修正三个点的位置 
				if (btm.pos._y > mid.pos._y) {
					tmp = mid;
					mid = btm;
					btm = tmp;
				}
				if (mid.pos._y > top.pos._y) {
					tmp = top;
					top = mid;
					mid = tmp;
				}
				if (btm.pos._y > mid.pos._y) {
					tmp = mid;
					mid = btm;
					btm = tmp;
				}

				triangle.top = top;
				triangle.mid = mid;
				triangle.btm = btm;

				if (type == SOLID) {
					triangle.numPixels = 0.5 * abs(
						(mid.pos._x - top.pos._x) * (btm.pos._y - top.pos._y) - (btm.pos._x - top.pos._x) * (mid.pos._y
							- top.pos._y));
				}
				else {
					triangle.numPixels = Distance(top.pos, mid.pos) + Distance(top.pos, btm.pos) + Distance(
						mid.pos, btm.pos);
				}

				triangles.push_back(triangle);
				numPixels += triangle.numPixels;
			}
		}


	}


	static void doRasterize(std::vector<Triangle>& triangles, std::vector<Pixel>& pixels, const TYPE type) {
		size_t index = 0;
		for (auto& triangle : triangles) {
			Raster::doRasterize(triangle, pixels, index, type);
		}
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
		auto numPixels = 0;
		doRenderFace(model , shader , type , triangles , numPixels);

		std::vector<Pixel> pixels(numPixels * 1.5);
		doRasterize(triangles, pixels, type);

		doRenderPixel(shader, pixels);

		pixels.clear();
		pixels.shrink_to_fit();

		triangles.clear();
		triangles.shrink_to_fit();
	}
};
