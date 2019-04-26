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
 * ����������
 */
class Camera {
public:
    /*
     * ��ƽ��
     */
    float near;
    /*
     * Զƽ��
     */
    float far;

    /*
     * eye ����
     */
    Math::Vector3 eye;

    /*
     * �չ��췽��
     */
    virtual ~Camera();

    /*
     * �����麯��
     */
    virtual Ray generateRay(float x, float y) = 0;
};

/*
 * ͸��ͶӰ�����
 */
class PerspectiveCamera : public Camera {
    /*
     * ����
     */
    float fovScale;
    /*
     * �������� up
     */
    Math::Vector3 up;
    /*
     * �������� right
     */
    Math::Vector3 right;
    /*
     * �������� front
     */
    Math::Vector3 front;

    /*
     * �չ��췽��
     */
    PerspectiveCamera();

public:
    /*
     * ���췽��
     */
    PerspectiveCamera(float fov, const Math::Vector3 &eye, const Math::Vector3 &lookAt, const Math::Vector3 &up,
                      float near, float far);

    /*
     * ��������
     */
    ~PerspectiveCamera();

    /*
     * ���� x y λ�����ɹ���
     */
    Ray generateRay(float x, float y);
};

/*
 * ����ͶӰ�����
 */
class OrthoCamera : public Camera {

};
