/**
 * File Name : ray.h
 * Author : Yang Fan
 * Date : 2018/4/26
 * define ray class
 */

#pragma once

#include "../includes/math/vector.hpp"

/*
 * 光线类
 */
class Ray {

public:
    /*
     * 光线起点 origin
     */
    Math::Vector3 origin;
    /*
     * 光线方向 direction
     */
    Math::Vector3 direction;

    /*
     * 空构造方法
     */
    Ray();

    /*
     * 值拷贝构造方法
     */
    Ray(const Math::Vector3 &origin, const Math::Vector3 &direction);

    /*
     * 对象拷贝构造方法
     */
    Ray(const Ray &r);

    /*
     * 重载 =
     */
    Ray &operator=(const Ray &r);
};