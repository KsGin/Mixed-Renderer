/**
 * File Name : main.cu
 * Author : Yang Fan
 * Date : 2018/11/27
 * program entrance
 */
	
#include "cuda/define.cu"
#include "common/device.cu"	

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
