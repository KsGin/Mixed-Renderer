/**
 * File Name : texture.h
 * Author : Yang Fan
 * Date : 2019/1/7
 * declare texture class
 */

#pragma once
#include <SDL_image.h>


/*
 * defined clamp
 */
#define CLAMP(x , min , max) { \
   if (x > max) x = max;  \
   if (x < min) x = min;  \
}

/*
 * defined clamp 0~1
 */
#define CLAMP01(x) { \
	CLAMP(x , 0 , 1) \
}

/*
 *	Texture class
 */
class Texture
{
public:

	/*
	 * width & height
	 */
	int width, height;

	/*
	 * pixels
	 */
	unsigned char* pixels;

	/*
	 * isAlpha
	 */
	bool alpha;

	Texture()
	{
		pixels = nullptr;
		width = 0;
		height = 0;
		alpha = false;
	}

	/*
	 * init from file
	 */
	static Texture LoadFromFile(const char* fileName, bool isAlpha)
	{
		Texture texture;

		IMG_Init(IMG_INIT_JPG | IMG_INIT_PNG);
		SDL_Surface* sf = IMG_Load(fileName);

		if (sf == nullptr)
		{
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
	                              bool isAlpha)
	{
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
	void setPixel(float x, float y, const Color& color)
	{
		const int tx = x * width;
		const int ty = y * height;

		auto idx = (ty * width + tx) * 4;

		CLAMP(idx, 0, width * height * 4 - 1);

		auto r = color.r;
		auto g = color.g;
		auto b = color.b;
		auto a = color.a;
		CLAMP01(a);
		CLAMP01(b);
		CLAMP01(g);
		CLAMP01(r);

		this->pixels[idx - 1] = static_cast<unsigned char>(a * 255);
		this->pixels[idx - 2] = static_cast<unsigned char>(b * 255);
		this->pixels[idx - 3] = static_cast<unsigned char>(g * 255);
		this->pixels[idx - 4] = static_cast<unsigned char>(r * 255);
	}


	/*
	 * get pixel
	 */
	void getPixel(float x, float y, Color& color)
	{
		const int tx = x * this->width;
		const int ty = y * this->height;

		auto idx = (ty * this->width + tx) * 4;

		CLAMP(idx, 0, this->width * this->height * 4 - 1);

		color.a = this->pixels[idx - 1] / 255.0f;
		color.b = this->pixels[idx - 2] / 255.0f;
		color.g = this->pixels[idx - 3] / 255.0f;
		color.r = this->pixels[idx - 4] / 255.0f;
	}
};