/**
 * File Name : main.cu
 * Author : Yang Fan
 * Date : 2018/11/27
 * program entrance
 */

#include "common/render.h"
#include "common/device.h"
#include "includes/math/matrix.hpp"

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

	auto model1 = Matrix::identity() * Matrix::scale(0.15 , 0.15 , 0.15);
	auto model2 = Matrix::identity() * Matrix::scale(0.15 , 0.15 , 0.15);

	auto rotation = Matrix::identity();
	auto view = Matrix::lookAtLH(Vector3(0, 0, 5), Vector3(0, 0, 0), Vector3(0, 1, 0));
	auto perspective = Matrix::perspectiveFovLH( 1 , SCREEN_WIDTH / SCREEN_HEIGHT, 0.1, 1000);

	auto cube1 = Model::cube();
	auto cube2 = Model::cube();
	// auto cube3 = Model::cube();

	auto shader = Shader();
	shader.setMat(view, VIEW);
	shader.setMat(perspective, PERSPECTIVE);

	auto texture = Texture::LoadFromFile("resources/TD1.png" , true);

	shader.setTexture(texture, 0);

	d.show();
	 
	while (!d.windowShouldClose()) {
		d.clear();

		rotation = rotation * Matrix::rotationY(-0.02f) * Matrix::rotationZ(-0.02f) * Matrix::rotationX(-0.02f);

		shader.setMat(model1 * rotation * Matrix::translate(-0.3 , 0 , 0), MODEL);
		Render::render(cube1, shader, SOLID);

		shader.setMat(model1 * rotation * Matrix::translate(0.3 , 0 , 0), MODEL);
		Render::render(cube2, shader, SOLID);
		
		// shader.setMat(model1 * rotation * Matrix::translate(  0 , 0 , -1), MODEL);
		// Render::render(cube3, shader, SOLID);

		d.handleEvent();
		d.updateRender();
	}

	d.destory();

	return 0;
}
