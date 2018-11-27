// Mixed-Renderer.cpp: 定义应用程序的入口点。
//

#include <iostream>
#include "Includes/common/device.h"		
#include "cuda/define.cu"

using namespace std;

int main()
{
	PRINT_DEVICE_INFORMATION();
	auto *d = new Device();
	d->initialize(SCREEN_WIDTH, SCREEN_HEIGHT, false);
	d->show();

	while (!d->windowShouldClose())
	{
		d->handleEvent();
		d->updateRender();
	}

	d->destory();
	delete d;

	return 0;
}
