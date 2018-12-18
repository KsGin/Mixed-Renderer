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
	auto* d = new Device();
	d->initialize(SCREEN_WIDTH, SCREEN_HEIGHT, IS_FULL_SCREEN, "Mixed-Renderer");
	d->show();

	auto* r = new Raster();
	r->initialize(d);

	while (!d->windowShouldClose()) {

		Matrix model = Matrix::identity();
		Matrix view = Matrix::lookAtLH(Vector3(0, 0, -2), Vector3(0, 0, 0), Vector3(0, 1, 0));
		Matrix perspective = Matrix::perspectiveFovLH(60, SCREEN_WIDTH / SCREEN_HEIGHT, 0.01, 1000);

		Matrix mvp = Matrix::transpose(model * view * perspective);

		Vector3 p1 = Matrix::transformCoordinates(Vector3(-1, 0, 0), mvp);
		Vector3 p2 = Matrix::transformCoordinates(Vector3(0, 2, 0), mvp);
		Vector3 p3 = Matrix::transformCoordinates(Vector3(1, 0, 0), mvp);

		// printf("%f %f %f\n", p2._x, p1._y, p1._z);

		r->draw(Vector2(p1._x, p1._y), Vector2(p2._x, p2._y), Vector2(p3._x, p3._y), Color::red() , Raster::SOLID);

		d->handleEvent();
		d->updateRender();
	}

	delete r;
	d->destory();
	delete d;

	return 0;
}
