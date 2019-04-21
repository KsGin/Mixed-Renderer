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
	auto model2 = Matrix::identity() * Matrix::scale(1.5 , 1 , 0.5);

	auto rotation = Matrix::identity();
	auto view = Matrix::lookAtLH(Vector3(0, 3, 5), Vector3(0, 1, 0), Vector3(0, 1, 0));
	auto perspective = Matrix::perspectiveFovLH( 1 , SCREEN_WIDTH / SCREEN_HEIGHT, 0.1, 1000);

	auto cube1 = Model::cube();
	auto floor = Model::floor();

	auto cubeShader = Shader();
	cubeShader.setMat(view, VIEW);
	cubeShader.setMat(perspective, PERSPECTIVE);

	auto floorShader = Shader();
	floorShader.setMat(view , VIEW);
	floorShader.setMat(perspective , PERSPECTIVE);

	cubeShader.setTexture(Texture::LoadFromFile("resources/TD2.png" , true), 0);
	floorShader.setTexture(Texture::LoadFromFile("resources/TD1.png" , true), 0);

	d.show();
	 
	while (!d.windowShouldClose()) {
		d.clear();

		rotation = rotation * Matrix::rotationY(-0.02f) * Matrix::rotationZ(-0.02f) * Matrix::rotationX(-0.02f);

		cubeShader.setMat(model1 * rotation * Matrix::translate(0 , 0.8 , 0), MODEL);
		Render::render(cube1, cubeShader, SOLID);
		
		floorShader.setMat(model2 * Matrix::translate(0 , 0 , 0.5), MODEL);
		Render::render(floor, floorShader, SOLID);

		d.handleEvent();
		d.updateRender();
	}

	d.destory();

	return 0;
}
