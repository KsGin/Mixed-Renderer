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


	static Model triangle() {
		Model triangle;
		Model::Mesh triangleMesh;
		Model::Mesh::Face triangleFace;

		triangleFace.v1.pos = Math::Vector3(0, 1, 0);
		triangleFace.v2.pos = Math::Vector3(-1, 0, 0);
		triangleFace.v3.pos = Math::Vector3(1, 0, 0);

		triangleFace.v1.normal = Math::Vector3(0, 0, -1);
		triangleFace.v2.normal = Math::Vector3(0, 0, -1);
		triangleFace.v3.normal = Math::Vector3(0, 0, -1);

		triangleFace.v1.color = Color::red();
		triangleFace.v2.color = Color::green();
		triangleFace.v3.color = Color::blue();

		triangleFace.v1.uv = Math::Vector2(0.5, 0);
		triangleFace.v2.uv = Math::Vector2(0, 1);
		triangleFace.v3.uv = Math::Vector2(1, 1);

		triangleMesh.faces.push_back(triangleFace);
		triangle.meshes.push_back(triangleMesh);

		return triangle;
	}

	static Model cube() {
		Model triangle;
		Model::Mesh triangleMesh;
		Model::Mesh::Face face1;

		face1.v1.pos = Math::Vector3(-1, -1, 0);
		face1.v2.pos = Math::Vector3(-1, 1, 0);
		face1.v3.pos = Math::Vector3(1, 1, 0);

		face1.v1.normal = Math::Vector3(0, 0, -1);
		face1.v2.normal = Math::Vector3(0, 0, -1);
		face1.v3.normal = Math::Vector3(0, 0, -1);

		face1.v1.color = Color::red();
		face1.v2.color = Color::green();
		face1.v3.color = Color::blue();

		face1.v1.uv = Math::Vector2(0, 1);
		face1.v2.uv = Math::Vector2(0, 0);
		face1.v3.uv = Math::Vector2(1, 0);

		triangleMesh.faces.push_back(face1);

		Model::Mesh::Face face2;

		face2.v1.pos = Math::Vector3(1, 1, 0);
		face2.v2.pos = Math::Vector3(1, -1, 0);
		face2.v3.pos = Math::Vector3(-1, -1, 0);
			
		face2.v1.normal = Math::Vector3(0, 0, -1);
		face2.v2.normal = Math::Vector3(0, 0, -1);
		face2.v3.normal = Math::Vector3(0, 0, -1);
			
		face2.v1.color = Color::blue();
		face2.v2.color = Color::green();
		face2.v3.color = Color::red();
			
		face2.v1.uv = Math::Vector2(1, 0);
		face2.v2.uv = Math::Vector2(1, 1);
		face2.v3.uv = Math::Vector2(0, 1);

		triangleMesh.faces.push_back(face2);

		triangle.meshes.push_back(triangleMesh);

		return triangle;
	}


private:

};

