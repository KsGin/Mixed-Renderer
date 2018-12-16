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
		// r->drawLine(Math::Vector2(100, 100), Math::Vector2(SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100),
		//             Color::blue());
		// r->drawLine(Math::Vector2(100, SCREEN_HEIGHT - 100), Math::Vector2(SCREEN_WIDTH - 100, 100), Color::red());

		r->drawTriangle(Math::Vector2(100, 100), Math::Vector2(200, 150), Math::Vector2(200, 200), Color::green());

		r->drawTriangle(Math::Vector2(300, 300), Math::Vector2(200, 400), Math::Vector2(400, 400), Color::green());

		r->drawTriangle(Math::Vector2(500, 100), Math::Vector2(400, 150), Math::Vector2(400, 200), Color::green());

		d->handleEvent();
		d->updateRender();
	}

	delete r;
	d->destory();
	delete d;

	return 0;
}
