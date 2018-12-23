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

	Matrix model = Matrix::identity();
	Matrix view = Matrix::lookAtLH(Vector3(0, 0, -2), Vector3(0, 0, 0), Vector3(0, 1, 0));
	Matrix perspective = Matrix::perspectiveFovLH(60, SCREEN_WIDTH / SCREEN_HEIGHT, 0.01, 1000);

	Model m = Model::cube();

	Shader shader;
	shader.setMat(view, Shader::MatType::VIEW);
	shader.setMat(perspective, Shader::MatType::PERSPECTIVE);

	d.show();

	while (!d.windowShouldClose()) {
		d.clear();

		model = model * Math::Matrix::rotationY(-0.01f);
		shader.setMat(model, Shader::MatType::MODEL);

		Render::render(m, shader, Raster::SOLID);

		d.handleEvent();
		d.updateRender();
	}

	d.destory();

	return 0;
}
