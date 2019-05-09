/**
 * File Name : main.cu
 * Author : Yang Fan
 * Date : 2018/11/27
 * program entrance
 */

#include "common/render.h"
#include "common/device.h"
#include "includes/math/matrix.hpp"
#include "common/camera.h"

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

	Renderer::initialize(NUM_PIXELS , NUM_COLORS , NUM_TRIANGLES , NUM_MODELS);
	auto r = Renderer::getInstance();

	auto model1 = Matrix::identity() * Matrix::scale(0.15 , 0.15 , 0.15);
	auto model2 = Matrix::identity() * Matrix::scale(1.5 , 1 , 0.5);

	auto camera = PerspectiveCamera(60 , Vector3(0, 4, 5) , Vector3(0, 1, 0) , Vector3(0, 1, 0) , 0.1 , 1000);
	r.setPerspectiveCamera(camera);

	auto rotation = Matrix::identity();
	auto view = Matrix::lookAtLH(camera.eye, camera.target, Vector3(0, 1, 0));
	auto perspective = Matrix::perspectiveFovLH( 1 , SCREEN_WIDTH / SCREEN_HEIGHT, camera.near, camera.far);

	auto cube1 = Model::cube();
	auto floor = Model::floor();

	auto cubeShader = Shader(CUBE);
	cubeShader.setMat(view, VIEW);
	cubeShader.setMat(perspective, PERSPECTIVE);

	auto waterShader = Shader(WATER);
	waterShader.setMat(view , VIEW);
	waterShader.setMat(perspective , PERSPECTIVE);

	cubeShader.setTexture(Texture::LoadFromFile("resources/cube.png" , true), 0);
	waterShader.setTexture(Texture::LoadFromFile("resources/water.png" , true), 0);

	d.show();

	Args args{0.0f};
	float bis = 0.001;
	while (!d.windowShouldClose()) {
		d.clear();
		r.clear();

		rotation = rotation * Matrix::rotationY(-0.02f) * Matrix::rotationZ(-0.02f) * Matrix::rotationX(-0.02f);

		cubeShader.setMat(model1 * rotation * Matrix::translate(0 , 0.8 , 0), MODEL);
		r.add(cube1 , cubeShader , SOLID);

		args.bis += bis;
		if (args.bis >= 0.1f || args.bis <= -0.1f) {bis = -bis;}
		waterShader.setArgs(args);

		waterShader.setMat(model2 * Matrix::translate(0 , 0 , 0.5), MODEL);
		r.add(floor , waterShader, SOLID);

		r.render();

		d.handleEvent();
		d.updateRender();
	}

	d.destory();
	r.destory();

	return 0;
}
