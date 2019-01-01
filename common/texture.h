/**
 * File Name : texture.h
 * Author : Yang Fan
 * Date : 2019/1/1
 * declare texture class
 */

#pragma once

#include <SDL_image.h>
#include "../cuda/define.cu"
#include "color.h"

/*
 *	Texture class
 */
class Texture {
	/*
	 * pixels
	 */
	unsigned char* pixels;

	/*
	 * width & height
	 */
	int width, height;

	/*
	 * isAlpha
	 */
	bool alpha;

public:

	Texture() {
		pixels = nullptr;
		width = 0;
		height = 0;
		alpha = false;
	}

	/*
	 * init from file
	 */
	static Texture LoadFromFile(const char* fileName, bool isAlpha) {
		Texture texture;

		IMG_Init(IMG_INIT_JPG | IMG_INIT_PNG);
		SDL_Surface* sf = IMG_Load(fileName);

		if (sf == nullptr) {
			fprintf(stderr, "could not load image: %s\n", IMG_GetError());
			return texture;
		}

		texture.pixels = static_cast<unsigned char *>(sf->pixels);
		texture.width = sf->w;
		texture.height = sf->h;
		texture.alpha = isAlpha;
		
		return texture;
	}

	/*
	 * init from memory
	 */
	static Texture LoadFromMemory(const int textureWidth, const int textureHeight, unsigned char* textureData,
	                              bool isAlpha) {
		Texture texture;
		texture.pixels = textureData;
		texture.width = textureWidth;
		texture.height = textureHeight;
		texture.alpha = isAlpha;
		return texture;
	}


	/*
	 * set pixel
	 */
	void setPixel(const float x, const float y, const Color& color) {
		const int tx = x * width;
		const int ty = y * height;

		auto idx = (ty * width + tx) * 4;

		CLAMP(idx, 0, width * height * 4 - 1);

		auto col = color;
		CLAMP01(col.a);
		CLAMP01(col.b);
		CLAMP01(col.g);
		CLAMP01(col.r);

		this->pixels[idx - 1] = static_cast<unsigned char>(col.a * 255);
		this->pixels[idx - 2] = static_cast<unsigned char>(col.b * 255);
		this->pixels[idx - 3] = static_cast<unsigned char>(col.g * 255);
		this->pixels[idx - 4] = static_cast<unsigned char>(col.r * 255);
	}


	/*
	 * get pixel
	 */
	Color getPixel(const float x, const float y) {

		const int tx = x * width;
		const int ty = y * height;

		auto idx = (ty * width + tx) * 4;

		CLAMP(idx, 0, width * height * 4 - 1);

		const auto a = this->pixels[idx - 1] / 255.0f;
		const auto b = this->pixels[idx - 2] / 255.0f;
		const auto g = this->pixels[idx - 3] / 255.0f;
		const auto r = this->pixels[idx - 4] / 255.0f;
		return Color(r, g, b, a);
	}
};
