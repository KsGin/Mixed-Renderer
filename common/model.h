/**
 * File Name : model.h
 * Author : Yang Fan
 * Date : 2018/12/19
 * model class
 */

#pragma once

#include <vector>


class Model
{
public:
	/*
     * Mesh
     */
	class Mesh {
	public:
		/*
	     * Face (triangle)
	     */
		class Face {
		public:
			/*
			 * Vertex
			 */
			class Vertex {
			public:
				Math::Vector3 pos;
				Math::Vector3 normal;
				Math::Vector2 uv;
				Color color;
			};
			Vertex v1, v2, v3;
		};

		std::vector<Face> faces;
	};

	std::vector<Mesh> meshes;

private:

};
