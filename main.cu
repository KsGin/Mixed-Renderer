// Mixed-Renderer.cpp: 定义应用程序的入口点。
//

#include <iostream>
#include "Includes/common/device.h"						

using namespace std;

#undef main 

int main()
{
	auto *d = new Device();
	d->initialize(800, 600, false);
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
