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
			Math::Vector3 p1, p2, p3;
			Math::Vector3 normal;
			Color color;
		};

		std::vector<Face> faces;
	};

	std::vector<Mesh> meshes;

private:

};
