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

		triangleFace.v1.pos = Math::Vector3( 0,  1, -5);
		triangleFace.v2.pos = Math::Vector3(-1,  0, -5);
		triangleFace.v3.pos = Math::Vector3( 1,  0, -5);

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

		face1.v1.pos = Math::Vector3(-1, -1, -1);
		face1.v2.pos = Math::Vector3(-1,  1, -1);
		face1.v3.pos = Math::Vector3( 1,  1, -1);

		face1.v1.normal = Math::Vector3(0, 0, -1);
		face1.v2.normal = Math::Vector3(0, 0, -1);
		face1.v3.normal = Math::Vector3(0, 0, -1);

		face1.v1.color = Color::red();
		face1.v2.color = Color::red();
		face1.v3.color = Color::red();

		face1.v1.uv = Math::Vector2(0, 1);
		face1.v2.uv = Math::Vector2(0, 0);
		face1.v3.uv = Math::Vector2(1, 0);

		triangleMesh.faces.push_back(face1);

		Model::Mesh::Face face2;

		face2.v1.pos = Math::Vector3( 1,  1, -1);
		face2.v2.pos = Math::Vector3( 1, -1, -1);
		face2.v3.pos = Math::Vector3(-1, -1, -1);
			
		face2.v1.normal = Math::Vector3(0, 0, -1);
		face2.v2.normal = Math::Vector3(0, 0, -1);
		face2.v3.normal = Math::Vector3(0, 0, -1);
			
		face2.v1.color = Color::red();
		face2.v2.color = Color::red();
		face2.v3.color = Color::red();
			
		face2.v1.uv = Math::Vector2(1, 0);
		face2.v2.uv = Math::Vector2(1, 1);
		face2.v3.uv = Math::Vector2(0, 1);

		triangleMesh.faces.push_back(face2);

		Model::Mesh::Face face3;

		face3.v1.pos = Math::Vector3(-1, 1, 1);
		face3.v2.pos = Math::Vector3( 1, 1, 1);
		face3.v3.pos = Math::Vector3(-1, 1, -1);

		face3.v1.normal = Math::Vector3(0, 1, 0);
		face3.v2.normal = Math::Vector3(0, 1, 0);
		face3.v3.normal = Math::Vector3(0, 1, 0);

		face3.v1.color = Color::green();
		face3.v2.color = Color::green();
		face3.v3.color = Color::green();

		face3.v1.uv = Math::Vector2(0, 0);
		face3.v2.uv = Math::Vector2(1, 0);
		face3.v3.uv = Math::Vector2(0, 1);

		triangleMesh.faces.push_back(face3);

		Model::Mesh::Face face4;

		face4.v1.pos = Math::Vector3( 1, 1,  1);
		face4.v2.pos = Math::Vector3( 1, 1, -1);
		face4.v3.pos = Math::Vector3(-1, 1, -1);

		face4.v1.normal = Math::Vector3(0, 1, 0);
		face4.v2.normal = Math::Vector3(0, 1, 0);
		face4.v3.normal = Math::Vector3(0, 1, 0);

		face4.v1.color = Color::green();
		face4.v2.color = Color::green();
		face4.v3.color = Color::green();

		face4.v1.uv = Math::Vector2(1, 0);
		face4.v2.uv = Math::Vector2(1, 1);
		face4.v3.uv = Math::Vector2(0, 1);

		triangleMesh.faces.push_back(face4);


		Model::Mesh::Face face5;

		face5.v1.pos = Math::Vector3(-1, -1,  1);
		face5.v2.pos = Math::Vector3(-1,  1,  1);
		face5.v3.pos = Math::Vector3(-1,  1, -1);

		face5.v1.normal = Math::Vector3(-1, 0, 0);
		face5.v2.normal = Math::Vector3(-1, 0, 0);
		face5.v3.normal = Math::Vector3(-1, 0, 0);

		face5.v1.color = Color::blue();
		face5.v2.color = Color::blue();
		face5.v3.color = Color::blue();

		face5.v1.uv = Math::Vector2(0, 1);
		face5.v2.uv = Math::Vector2(0, 0);
		face5.v3.uv = Math::Vector2(1, 0);

		triangleMesh.faces.push_back(face5);

		Model::Mesh::Face face6;

		face6.v1.pos = Math::Vector3(-1, -1,  1);
		face6.v2.pos = Math::Vector3(-1,  1, -1);
		face6.v3.pos = Math::Vector3(-1, -1, -1);

		face6.v1.normal = Math::Vector3(-1, 0, 0);
		face6.v2.normal = Math::Vector3(-1, 0, 0);
		face6.v3.normal = Math::Vector3(-1, 0, 0);

		face6.v1.color = Color::blue();
		face6.v2.color = Color::blue();
		face6.v3.color = Color::blue();

		face6.v1.uv = Math::Vector2(0, 1);
		face6.v2.uv = Math::Vector2(1, 0);
		face6.v3.uv = Math::Vector2(1, 1);

		triangleMesh.faces.push_back(face6);


		Model::Mesh::Face face7;

		face7.v1.pos = Math::Vector3(-1, -1, 1);
		face7.v2.pos = Math::Vector3(-1,  1, 1);
		face7.v3.pos = Math::Vector3( 1,  1, 1);

		face7.v1.normal = Math::Vector3(0, 0, 1);
		face7.v2.normal = Math::Vector3(0, 0, 1);
		face7.v3.normal = Math::Vector3(0, 0, 1);

		face7.v1.color = Color::red();
		face7.v2.color = Color::red();
		face7.v3.color = Color::red();

		face7.v1.uv = Math::Vector2(1, 1);
		face7.v2.uv = Math::Vector2(1, 0);
		face7.v3.uv = Math::Vector2(0, 0);

		triangleMesh.faces.push_back(face7);

		Model::Mesh::Face face8;

		face8.v1.pos = Math::Vector3( 1,  1, 1);
		face8.v2.pos = Math::Vector3( 1, -1, 1);
		face8.v3.pos = Math::Vector3(-1, -1, 1);

		face8.v1.normal = Math::Vector3(0, 0, 1);
		face8.v2.normal = Math::Vector3(0, 0, 1);
		face8.v3.normal = Math::Vector3(0, 0, 1);

		face8.v1.color = Color::red();
		face8.v2.color = Color::red();
		face8.v3.color = Color::red();

		face8.v1.uv = Math::Vector2(0, 0);
		face8.v2.uv = Math::Vector2(0, 1);
		face8.v3.uv = Math::Vector2(1, 1);

		triangleMesh.faces.push_back(face8);

		Model::Mesh::Face face9;

		face9.v1.pos = Math::Vector3(-1, -1,  1);
		face9.v2.pos = Math::Vector3( 1, -1,  1);
		face9.v3.pos = Math::Vector3(-1, -1, -1);

		face9.v1.normal = Math::Vector3(0, -1, 0);
		face9.v2.normal = Math::Vector3(0, -1, 0);
		face9.v3.normal = Math::Vector3(0, -1, 0);

		face9.v1.color = Color::green();
		face9.v2.color = Color::green();
		face9.v3.color = Color::green();

		face9.v1.uv = Math::Vector2(1, 0);
		face9.v2.uv = Math::Vector2(0, 0);
		face9.v3.uv = Math::Vector2(1, 1);

		triangleMesh.faces.push_back(face9);

		Model::Mesh::Face face10;

		face10.v1.pos = Math::Vector3(1,  -1, 1);
		face10.v2.pos = Math::Vector3(1,  -1, -1);
		face10.v3.pos = Math::Vector3(-1, -1, -1);

		face10.v1.normal = Math::Vector3(0, -1, 0);
		face10.v2.normal = Math::Vector3(0, -1, 0);
		face10.v3.normal = Math::Vector3(0, -1, 0);

		face10.v1.color = Color::green();
		face10.v2.color = Color::green();
		face10.v3.color = Color::green();

		face10.v1.uv = Math::Vector2(0, 0);
		face10.v2.uv = Math::Vector2(0, 1);
		face10.v3.uv = Math::Vector2(1, 1);

		triangleMesh.faces.push_back(face10);

		Model::Mesh::Face face11;

		face11.v1.pos = Math::Vector3(1,-1, 1);
		face11.v2.pos = Math::Vector3(1, 1, 1);
		face11.v3.pos = Math::Vector3(1, 1, -1);

		face11.v1.normal = Math::Vector3(1, 0, 0);
		face11.v2.normal = Math::Vector3(1, 0, 0);
		face11.v3.normal = Math::Vector3(1, 0, 0);

		face11.v1.color = Color::blue();
		face11.v2.color = Color::blue();
		face11.v3.color = Color::blue();

		face11.v1.uv = Math::Vector2(1, 1);
		face11.v2.uv = Math::Vector2(1, 0);
		face11.v3.uv = Math::Vector2(-1, 0);

		triangleMesh.faces.push_back(face11);

		Model::Mesh::Face face12;

		face12.v1.pos = Math::Vector3(1, -1, 1);
		face12.v2.pos = Math::Vector3(1, 1, -1);
		face12.v3.pos = Math::Vector3(1, -1, -1);

		face12.v1.normal = Math::Vector3(1, 0, 0);
		face12.v2.normal = Math::Vector3(1, 0, 0);
		face12.v3.normal = Math::Vector3(1, 0, 0);

		face12.v1.color = Color::blue();
		face12.v2.color = Color::blue();
		face12.v3.color = Color::blue();

		face12.v1.uv = Math::Vector2(1, 1);
		face12.v2.uv = Math::Vector2(0, 0);
		face12.v3.uv = Math::Vector2(0, 1);

		triangleMesh.faces.push_back(face12);

		triangle.meshes.push_back(triangleMesh);

		return triangle;
	}


private:

};

