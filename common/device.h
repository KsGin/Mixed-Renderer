/**
 * File Name : device.h
 * Author : Yang Fan
 * Date : 2018/11/27
 * device class
 */

#pragma once

#include <SDL.h>
#include <memory.h>
#include <ctime>
#include "color.h"

/*
 * 设备类
 */
class Device {
	/*
	 * window 指针
	 */
	SDL_Window* window;
	/*
	 * texture 指针
	 */
	SDL_Texture* texture;
	/*
	 * renderer 指针
	 */
	SDL_Renderer* renderer;

	/*
	 * pixels 指针
	 */
	Uint8* pixels;

	/*
	 * 是否退出
	 */
	bool isQuit;

	/*
	 * SDL 事件
	 */
	SDL_Event event;

	/*
	 * 空构造方法
	 */
	Device() {
		this->pixels = 0;
		this->window = 0;
		this->renderer = 0;
		this->texture = 0;
	}

public:

	/*
	 * 窗口属性 宽 高
	 */
	int width, height;

	/*
	 * 是否全屏
	 */
	bool isScreenFull;

	/*
	 * 析构方法
	 */
	~Device() {
		//if (this->pixels) {
		//	delete[] this->pixels;
		//	this->pixels = 0;
		//}
	}

	/*
	 * 单例
	 */
	static Device& getInstance() {
		static Device instance;
		return instance;
	}

	/*
	 * 初始化设备
	 */
	static bool initialize(int width, int height, bool isScreenFull, const char* title) {

		auto &d = getInstance();

		d.width = width;
		d.height = height;
		d.isScreenFull = isScreenFull;

		if (SDL_Init(SDL_INIT_EVERYTHING) == -1) { return false; }

		Uint32 windowAttr = SDL_WINDOW_RESIZABLE;
		if (d.isScreenFull) { windowAttr |= SDL_WINDOW_FULLSCREEN_DESKTOP; }

		d.window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
			d.width, d.height, windowAttr);
		if (!d.window) { return false; }

		if (d.isScreenFull) { SDL_GetWindowSize(d.window, &d.width, &d.height); }

		d.renderer = SDL_CreateRenderer(d.window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
		if (!d.renderer) { return false; }

		d.pixels = new Uint8[d.width * d.height * 4];
		memset(d.pixels, 0, static_cast<size_t>(d.width * d.height * 4));
		if (!d.pixels) { return false; }

		d.texture = SDL_CreateTexture(d.renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, d.width,
			d.height);
		if (!d.texture) { return false; }

		d.isQuit = false;
		return true;
	}

	void clear() {
		if (pixels != nullptr) {
			memset(pixels, 0, width * height * 4);
		}
	}

	/*
	 * 显示窗口
	 */
	void show() { SDL_ShowWindow(window); }

	/*
	 * 销毁窗口
	 */
	void destory() {
		SDL_DestroyTexture(texture);
		SDL_DestroyRenderer(renderer);
		SDL_DestroyWindow(window);
		SDL_Quit();
	}

	/*
	 * 更新渲染
	 */
	void updateRender() {
		updatePixelsColor();
		SDL_RenderClear(renderer);
		SDL_RenderCopy(renderer, texture, nullptr, nullptr);
		SDL_RenderPresent(renderer);
	}

	/*
	 * 是否退出
	 */
	bool windowShouldClose() { return isQuit; }

	/*
	 * 修改像素颜色
	 */
	void setPixelColor(const int x, const int y, const Color& color) {
		int idx = (y * width + x) * 4;

		this->pixels[idx - 1] = (Uint8)(clamp(color.a) * 255);
		this->pixels[idx - 2] = (Uint8)(clamp(color.b) * 255);
		this->pixels[idx - 3] = (Uint8)(clamp(color.g) * 255);
		this->pixels[idx - 4] = (Uint8)(clamp(color.r) * 255);
	}

	/*
	 * 获得像素颜色
	 */
	Color getPixelColor(const int x, const int y) {

		const int tx = clamp(x, 0, width);
		const int ty = clamp(y, 0, height);

		const auto idx = (ty * width + tx) * 4;

		const auto a = this->pixels[idx - 1] / 255.0f;
		const auto b = this->pixels[idx - 2] / 255.0f;
		const auto g = this->pixels[idx - 3] / 255.0f;
		const auto r = this->pixels[idx - 4] / 255.0f;
		return Color(r, g, b, a);
	}

	/*
	 * 更新像素数据
	 */
	void updatePixelsColor() { SDL_UpdateTexture(this->texture, nullptr, this->pixels, width * 4); }

	/*
	 * 更新窗口标题
	 */
	void updateWindowTitle(const char* title) { SDL_SetWindowTitle(this->window, title); }

	/*
	 * 事件处理
	 */
	void handleEvent() { while (SDL_PollEvent(&event)) { if (event.type == SDL_QUIT) isQuit = true; } }

	/*
	 * 数值限制
	 */
	static float clamp(const float& value, const float& maxValue = 1, const float& minValue = 0) {
		float ret = value > minValue ? value : minValue;
		return ret < maxValue ? ret : maxValue;
	}
};
