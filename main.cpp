/**
 * File Name : main.cu
 * Author : Yang Fan
 * Date : 2018/11/27
 * program entrance
 */

#include "cuda/define.cu"
#include "common/device.h"
#include "common/raster.h"

using namespace std;

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

		r->draw(Math::Vector2(-0.2, 0.4), Math::Vector2(-0.4, 0), Math::Vector2(0, 0), Color::green() , Raster::WIREFRAME);

		d->handleEvent();
		d->updateRender();
	}

	delete r;
	d->destory();
	delete d;

	return 0;
}
