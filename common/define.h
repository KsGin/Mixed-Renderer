/**
 * File Name : define.h
 * Author : Yang Fan
 * Date : 2018/11/27
 * define some functions
 */

#pragma once


#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <SDL.h>
#include <ctime>
#include "../includes/math/vector.hpp"
#include "../includes/color.hpp"
#include "../includes/math/matrix.hpp"
#include "texture.h"
#include <vector>



/**
 * define a cuda call
 */
#define CUDA_CALL(x) { \
	const cudaError_t e = (x);\
	if(e != cudaSuccess)		\
	{		\
		printf("\nCUDA Error : %s (err_num = %d)\n", cudaGetErrorString(e), e); \
		cudaDeviceReset(); \
		assert(0); \
	}\
}

/**
 * define screen width & height & isFullScreen
 */
#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600
#define IS_FULL_SCREEN false

/*
 * define default pixels & colors & triangles & model count
 */
#define NUM_PIXELS 25600
#define NUM_COLORS 25600
#define NUM_TRIANGLES 256
#define NUM_MODELS 16
#define NUM_RAYS (SCREEN_HEIGHT * SCREEN_WIDTH)

/*
 * print device information 
 */
#define PRINT_DEVICE_INFORMATION() { \
    int dc;   \
	cudaGetDeviceCount(&dc); \
	if (dc == 0) { \
		printf("error : no device supporting cuda\n"); \
		exit(1); \
	} \
		\
	int dev = 0;\
	cudaSetDevice(dev);	\
	cudaDeviceProp devProps;	\
	cudaGetDeviceProperties(&devProps, dev); \
	\
	printf("name : %s \ntotalGlobalMem : %zdM\n" , devProps.name , devProps.totalGlobalMem >> 20); \
}

/*
 * INTERPOLATE FLOAT VALUE DEFINED
 */
#define INTERPOLATE(a , b , g , r) {										\
	CLAMP01(g);																\
	int d = a > b;															\
	r = d * (a - (a - b) * g) + (1-d) * (a + (b - a) * g);					\
}

/*
 * INTERPOLATE VECTOR2 VALUE DEFINED
 */
#define INTERPOLATEV2(v1 , v2 , gad , result) {								\
	INTERPOLATE(v1._x, v2._x, gad , result._x);								\
	INTERPOLATE(v1._y, v2._y, gad , result._y);								\
}

/*
 * INTERPOLATE VECTOR3 VALUE DEFINED
 */
#define INTERPOLATEV3(v1 , v2 , gad , result) {								\
	INTERPOLATE(v1._x, v2._x, gad , result._x);								\
	INTERPOLATE(v1._y, v2._y, gad , result._y);								\
	INTERPOLATE(v1._z, v2._z, gad , result._z);								\
}

#define INTERPOLATEC(v1 , v2 , gad , result) {								\
	INTERPOLATE(v1.r, v2.r, gad , result.r);								\
	INTERPOLATE(v1.g, v2.g, gad , result.g);								\
	INTERPOLATE(v1.b, v2.b, gad , result.b);								\
	INTERPOLATE(v1.a, v2.a, gad , result.a);								\
}

#define INTERPOLATEP(p1 , p2 , gad , result) {								\
	INTERPOLATEV3(p1.pos , p2.pos , gad , result.pos);						\
	INTERPOLATEV3(p1.pos3D , p2.pos3D , gad , result.pos3D);				\
	INTERPOLATEV3(p1.normal , p2.normal , gad , result.normal);				\
	INTERPOLATEV2(p1.uv , p2.uv , gad , result.uv);							\
	INTERPOLATEC(p1.color , p2.color , gad , result.color);					\
	INTERPOLATE(p1.reflectiveness, p2.reflectiveness, gad , result.reflectiveness);	\
	result.sType = p1.sType;												\
}


/*
 * Shader Matrix Type
 */
enum MatType {
	MODEL,
	VIEW,
	PERSPECTIVE
};

/*
 * Draw Type
 */
enum RenderType {
	SOLID,
	WIREFRAME
};


/*
 * 定义多个Shader
 */
enum ShaderType {
	WATER,
	CUBE ,
	LIGHT
};

/*
 * defined Vertex & Pixel
 */
typedef struct Vertex {
	Math::Vector3 pos;
	Math::Vector3 pos3D;
	Math::Vector3 normal;
	Math::Vector2 uv;
	Color color;
	float reflectiveness;
	ShaderType sType;
} Pixel;


/*
 * 定义三角形数据结构
 */
struct Triangle {
	/*
	 * 三个顶点
	 */
	Pixel top, mid, btm;
	/*
	 * 像素个数
	 */
	int numPixels;
};

/*
 * 定义线段数据结构
 */
struct Line {
	/*
	 * 两个顶点
	 */
	Pixel left, right;

	/*
	 * 像素个数
	 */
	int numPixels;
};

/*
 * 定义 Shader 多余参数
 */
struct Args {
	float bis;
};

struct ShaderData {
	/*
	 * modelMat
	 */
	Math::Matrix modelMat = Math::Matrix::identity();
	/*
	 * viewMat
	 */
	Math::Matrix viewMat = Math::Matrix::identity();
	/*
	 * perspectiveMat
	 */
	Math::Matrix perspectiveMat = Math::Matrix::identity();
	/*
	 * textures
	 */
	std::vector<Texture> textures = std::vector<Texture>(8);
	/*
	 * args
	 */
	Args args;

	/*
	 * shader type
	 */
	ShaderType sType;
};

/*
 * 相交检测结果
 */
struct IntersectResult {
	bool isSucceed;
	float distance;
	Math::Vector3 intersectPoint;
};

/*
 * Distance of two points
 */
static int Distance(const Math::Vector3& p1, const Math::Vector3& p2) {
	return sqrt(pow(p2._x - p1._x, 2) + pow(p2._y - p1._y, 2));
}
