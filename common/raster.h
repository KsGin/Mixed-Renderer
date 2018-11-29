/**
 * File Name : raster.cu
 * Author : Yang Fan
 * Date : 2018/11/28
 * declare raster
 */

#pragma once

#include "../includes/math/vector.hpp"
#include "color.h"
#include "device.h"
#include <cmath>

class Raster
{
private:
	Device *device;

public:
	/**
	 * 构造方法
	 */
	Raster() {}

	/*
	 * 析构方法
	 */
	~Raster()
	{
		if (device)
		{
			delete device;
			device = 0;
		}
	}

	void initialize(Device *device)
	{
		this->device = device;
	}

	void BresenhamRasterLine(const Math::Vector2 &p1 , const Math::Vector2 &p2 , const Color &color)
	{
		int x1 = p1._x , x2 = p2._x , y1 = p1._y , y2 = p2._y;
		const int dx = abs(x2 - x1);
		const int dy = abs(y2 - y1);
		int sx = x1 < x2 ? 1 : -1;
		int	sy = y1 < y2 ? 1 : -1;
		int	eps = dx - dy;
		while (true)
		{
			device->setPixelColor(x1, y1, color);
			if (x1 == x2 && y1 == y2) break;
			auto e2 = 2 * eps;
			if (e2 > -dy)
			{
				eps -= dy;
				x1 += sx;
			}

			if (e2 < dx)
			{
				eps += dx;
				y1 += sy;
			}
		}
	}
};
