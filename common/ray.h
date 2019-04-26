/**
 * File Name : ray.h
 * Author : Yang Fan
 * Date : 2018/4/26
 * define ray class
 */

#pragma once

#include "../includes/math/vector.hpp"

/*
 * ������
 */
class Ray {

public:
    /*
     * ������� origin
     */
    Math::Vector3 origin;
    /*
     * ���߷��� direction
     */
    Math::Vector3 direction;

    /*
     * �չ��췽��
     */
    Ray();

    /*
     * ֵ�������췽��
     */
    Ray(const Math::Vector3 &origin, const Math::Vector3 &direction);

    /*
     * ���󿽱����췽��
     */
    Ray(const Ray &r);

    /*
     * ���� =
     */
    Ray &operator=(const Ray &r);
};