/**
 * File Name : camera.h
 * Author : Yang Fan
 * Date : 2018/11/27
 * declare camera property and method
 */

#pragma once

#include "define.h"
#include "../includes/math/vector.hpp"
#include "ray.h"

/*
 * 摄像机类基类
 */
class Camera {
public:
    /*
     * 近平面
     */
    float near;
    /*
     * 远平面
     */
    float far;

    /*
     * eye 坐标
     */
    Math::Vector3 eye;

    /*
     * 空构造方法
     */
    virtual ~Camera();

    /*
     * 定义虚函数
     */
    virtual Ray generateRay(float x, float y) = 0;
};

/*
 * 透视投影摄像机
 */
class PerspectiveCamera : public Camera {
    /*
     * 缩放
     */
    float fovScale;
    /*
     * 方向向量 up
     */
    Math::Vector3 up;
    /*
     * 方向向量 right
     */
    Math::Vector3 right;
    /*
     * 方向向量 front
     */
    Math::Vector3 front;

    /*
     * 空构造方法
     */
    PerspectiveCamera();

public:
    /*
     * 构造方法
     */
    PerspectiveCamera(float fov, const Math::Vector3 &eye, const Math::Vector3 &lookAt, const Math::Vector3 &up,
                      float near, float far);

    /*
     * 析构方法
     */
    ~PerspectiveCamera();

    /*
     * 根据 x y 位置生成光线
     */
    Ray generateRay(float x, float y);
};

/*
 * 正交投影摄像机
 */
class OrthoCamera : public Camera {

};
