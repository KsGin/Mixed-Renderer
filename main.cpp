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

	Renderer::initialize(NUM_PIXELS , NUM_COLORS , NUM_TRIANGLES);
	auto r = Renderer::getInstance();

	auto cubeModel1 = Matrix::identity() * Matrix::scale(0.1 , 0.1 , 0.1);
	auto cubeModel2 = Matrix::identity() * Matrix::scale(0.1 , 0.1 , 0.1);
	auto waterModel = Matrix::identity() * Matrix::scale(1.5 , 1 , 1.5);
	auto lightModel =  Matrix::identity() * Matrix::scale( 0.02 , 0.02 , 0.02);

	auto camera = PerspectiveCamera(60 , Vector3(0, 5, 5) , Vector3(0, 0, 0) , Vector3(0, 1, 0) , 0.1 , 100);
	r.setPerspectiveCamera(camera);

	auto rotation1 = Matrix::identity();
	auto rotation2 = Matrix::identity();

	auto view = Matrix::lookAtLH(camera.eye, camera.target, Vector3(0, 1, 0));
	auto perspective = Matrix::perspectiveFovLH( 1 , SCREEN_WIDTH / SCREEN_HEIGHT, camera.near, camera.far);

	auto cube1 = Model::cube();
	auto cube2 = Model::cube();
	auto floor = Model::floor();
	auto light = Model::cube();

	auto cubeShader = Shader(CUBE);
	cubeShader.setMat(view, VIEW);
	cubeShader.setMat(perspective, PERSPECTIVE);

	auto floorShader = Shader(WATER);
	floorShader.setMat(view , VIEW);
	floorShader.setMat(perspective , PERSPECTIVE);
	
	auto lightShader = Shader(LIGHT);
	lightShader.setMat(view ,VIEW);
	lightShader.setMat(perspective , PERSPECTIVE);
	//
	// cubeShader.setTexture(Texture::LoadFromFile("resources/cube.png" , true), 0);
	// waterShader.setTexture(Texture::LoadFromFile("resources/water.png" , true), 0);

	d.show();

	Args args{0.0f};
	float bis = 0.0008f;
	while (!d.windowShouldClose()) {
		d.clear();
		r.clear();

		rotation1 = rotation1 * Matrix::rotationY(-0.02f) * Matrix::rotationZ(-0.02f) * Matrix::rotationX(-0.02f);
		rotation2 = rotation2 * Matrix::rotationY( 0.02f) * Matrix::rotationZ( 0.02f) * Matrix::rotationX( 0.02f);


		cubeShader.setMat(cubeModel1 * rotation1 * Matrix::translate(0.6 , 1 , 0), MODEL);
		r.add(cube1 , cubeShader , 0 , SOLID);

		cubeShader.setMat(cubeModel2 * rotation2 * Matrix::translate(-0.6 , 1 , 0), MODEL);
		r.add(cube2 , cubeShader , 0 , SOLID);

		// args.bis += bis;
		// if (args.bis >= 0.1f || args.bis <= -0.1f) {bis = -bis;}
		// waterShader.setArgs(args);

		floorShader.setMat(waterModel * Matrix::translate(0 , 0 , 0.5), MODEL);
		r.add(floor , floorShader ,0.5 , SOLID);

		//
		lightShader.setMat(lightModel * rotation1 * Matrix::translate(0 , 2 , 0) , MODEL);
		r.add(light , lightShader , 0 , SOLID);

		r.render();

		d.handleEvent();
		d.updateRender();
	}

	d.destory();
	r.destory();

	return 0;
}
