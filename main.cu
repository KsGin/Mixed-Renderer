// Mixed-Renderer.cpp: ����Ӧ�ó������ڵ㡣
//

#include <iostream>
#include "common/device.cu"		
#include "cuda/define.cu"

using namespace std;

int main()
{
	PRINT_DEVICE_INFORMATION();
	auto *d = new Device();
	d->initialize(SCREEN_WIDTH, SCREEN_HEIGHT, IS_FULL_SCREEN);
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
