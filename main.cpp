/**
 * File Name : main.cu
 * Author : Yang Fan
 * Date : 2018/11/27
 * program entrance
 */

#include "cuda/define.cu"
#include "common/device.h"
#include "common/raster.h"
#include "includes/math/matrix.hpp"
#include "common/render.h"

using namespace std;
using namespace Math;

/**
 * sdl defined main to SDL_main , so we undef it on here
 */
#undef main
int main()
/*
 * redefine main
 */
#define main SDL_main
{
	PRINT_DEVICE_INFORMATION();

	Device::initialize(SCREEN_WIDTH, SCREEN_HEIGHT, IS_FULL_SCREEN, "Mixed-Renderer");
	auto d = Device::getInstance();

	d.show();

	Matrix model = Matrix::identity();
	Matrix view = Matrix::lookAtLH(Vector3(0, 0, -15), Vector3(0, 0, 0), Vector3(0, 1, 0));
	Matrix perspective = Matrix::perspectiveFovLH(60, SCREEN_WIDTH / SCREEN_HEIGHT, 0.01, 1000);

	Model::Mesh::Face face1;
	face1.p1 = Vector3(-1, -1, 0);
	face1.p2 = Vector3(-1,  1, 0);
	face1.p3 = Vector3( 1,  1, 0);
	face1.normal = Vector3(0, 0, 1);
	face1.color = Color::red();

	Model::Mesh::Face face2;
	face2.p1 = Vector3(-1, -1, 0);
	face2.p2 = Vector3(1, 1, 0);
	face2.p3 = Vector3(1, -1, 0);
	face2.normal = Vector3(0, 0, 1);
	face2.color = Color::red();

	Model::Mesh mesh;
	mesh.faces.push_back(face1);
	mesh.faces.push_back(face2);

	Model m;
	m.meshes.push_back(mesh);

	Shader shader;
	shader.setMat(model , Shader::MatType::MODEL);
	shader.setMat(view, Shader::MatType::VIEW);
	shader.setMat(perspective, Shader::MatType::PERSPECTIVE);

	while (!d.windowShouldClose()) {
		d.clear();

		model = model * Math::Matrix::rotationY(-0.01);
		shader.setMat(model, Shader::MatType::MODEL);

		Render::render(m, shader, Raster::SOLID);

		d.handleEvent();
		d.updateRender();
	}

	d.destory();

	return 0;
}
