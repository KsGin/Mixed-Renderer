/*
 * File   : matrix
 * Author : KsGin
 * Date   : 2018/5/23
 */

#ifndef MATHLIB_MATRIX_HPP
#define MATHLIB_MATRIX_HPP

#include <cmath>

#include "vector.hpp"

namespace Math {

    /**
     * 矩阵类
     */
    class Matrix {

        /*
         * 空构造函数 禁止使用 禁止重写
         */
        Matrix() {}

    public:
        /*
         * 数据定义
         */
        union {
            float _m[16];
            struct {
                float _11, _12, _13, _14;
                float _21, _22, _23, _24;
                float _31, _32, _33, _34;
                float _41, _42, _43, _44;
            };
        };


        /*
         * 值构造函数
         */
        Matrix(float m1, float m2, float m3, float m4,
               float m5, float m6, float m7, float m8,
               float m9, float m10, float m11, float m12,
               float m13, float m14, float m15, float m16) {
            _m[0] = m1;
            _m[1] = m2;
            _m[2] = m3;
            _m[3] = m4;
            _m[4] = m5;
            _m[5] = m6;
            _m[6] = m7;
            _m[7] = m8;
            _m[8] = m9;
            _m[9] = m10;
            _m[10] = m11;
            _m[11] = m12;
            _m[12] = m13;
            _m[13] = m14;
            _m[14] = m15;
            _m[15] = m16;
        }

        /*
         * 引用构造函数
         */
        Matrix(const Matrix &mat) {
            for (int i = 0; i < 16; ++i)
                _m[i] = mat._m[i];
        }

        /*
         * 析构函数
         */
        ~Matrix() {
        }


        /*
         * 重载 =
         */
        Matrix& operator=(Matrix const &mat) = default;

        /*
         * 重载 ==
         */
        bool operator==(const Matrix &mat) const {
            for (int i = 0; i < 16; ++i) {
                if (mat._m[i] != _m[i]) return false;
            }
            return true;
        }

        /*
         * 重载 +
         */
        Matrix operator+(const Matrix &mat) const {
            float tm[16];
            for (int i = 0; i < 16; ++i) {
                tm[i] = _m[i] + mat._m[i];
            }
            return Matrix(
                    tm[0], tm[1], tm[2], tm[3],
                    tm[4], tm[5], tm[6], tm[7],
                    tm[8], tm[9], tm[10], tm[11],
                    tm[12], tm[13], tm[14], tm[15]
            );
        }

        /*
         * 重载 -
         */
        Matrix operator-(const Matrix &mat) const {
            float tm[16];
            for (int i = 0; i < 16; ++i) {
                tm[i] = _m[i] - mat._m[i];
            }
            return Matrix(
                    tm[0], tm[1], tm[2], tm[3],
                    tm[4], tm[5], tm[6], tm[7],
                    tm[8], tm[9], tm[10], tm[11],
                    tm[12], tm[13], tm[14], tm[15]
            );
        }

        /*
         * 矩阵与矩阵相乘
         */
        Matrix multiply(const Matrix &mat) const {
            float tm[16];
            tm[0] = _m[0] * mat._m[0] + _m[1] * mat._m[4] + _m[2] * mat._m[8] + _m[3] * mat._m[12];
            tm[1] = _m[0] * mat._m[1] + _m[1] * mat._m[5] + _m[2] * mat._m[9] + _m[3] * mat._m[13];
            tm[2] = _m[0] * mat._m[2] + _m[1] * mat._m[6] + _m[2] * mat._m[10] + _m[3] * mat._m[14];
            tm[3] = _m[0] * mat._m[3] + _m[1] * mat._m[7] + _m[2] * mat._m[11] + _m[3] * mat._m[15];
            tm[4] = _m[4] * mat._m[0] + _m[5] * mat._m[4] + _m[6] * mat._m[8] + _m[7] * mat._m[12];
            tm[5] = _m[4] * mat._m[1] + _m[5] * mat._m[5] + _m[6] * mat._m[9] + _m[7] * mat._m[13];
            tm[6] = _m[4] * mat._m[2] + _m[5] * mat._m[6] + _m[6] * mat._m[10] + _m[7] * mat._m[14];
            tm[7] = _m[4] * mat._m[3] + _m[5] * mat._m[7] + _m[6] * mat._m[11] + _m[7] * mat._m[15];
            tm[8] = _m[8] * mat._m[0] + _m[9] * mat._m[4] + _m[10] * mat._m[8] + _m[11] * mat._m[12];
            tm[9] = _m[8] * mat._m[1] + _m[9] * mat._m[5] + _m[10] * mat._m[9] + _m[11] * mat._m[13];
            tm[10] = _m[8] * mat._m[2] + _m[9] * mat._m[6] + _m[10] * mat._m[10] + _m[11] * mat._m[14];
            tm[11] = _m[8] * mat._m[3] + _m[9] * mat._m[7] + _m[10] * mat._m[11] + _m[11] * mat._m[15];
            tm[12] = _m[12] * mat._m[0] + _m[13] * mat._m[4] + _m[14] * mat._m[8] + _m[15] * mat._m[12];
            tm[13] = _m[12] * mat._m[1] + _m[13] * mat._m[5] + _m[14] * mat._m[9] + _m[15] * mat._m[13];
            tm[14] = _m[12] * mat._m[2] + _m[13] * mat._m[6] + _m[14] * mat._m[10] + _m[15] * mat._m[14];
            tm[15] = _m[12] * mat._m[3] + _m[13] * mat._m[7] + _m[14] * mat._m[11] + _m[15] * mat._m[15];
            return Matrix(
                    tm[0], tm[1], tm[2], tm[3],
                    tm[4], tm[5], tm[6], tm[7],
                    tm[8], tm[9], tm[10], tm[11],
                    tm[12], tm[13], tm[14], tm[15]
            );
        }

        /*
         * 重载 *
         */
        Matrix operator*(const Matrix &mat) {
            return multiply(mat);
        }

        /*
         * 获得零矩阵
         */
        static Matrix zero() {
            return Matrix(0, 0, 0, 0,
                          0, 0, 0, 0,
                          0, 0, 0, 0,
                          0, 0, 0, 0);
        }

        /*
         *  获得单位矩阵
         */
        static Matrix identity() {
            return Matrix(
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1);
        }

        /*
         * 获得缩放矩阵
         */
        static Matrix scale(float x, float y, float z) {
            return Matrix(
                    x, 0, 0, 0,
                    0, y, 0, 0,
                    0, 0, z, 0,
                    0, 0, 0, 1
            );
        }

        /*
         * 获得平移矩阵
         */
        static Matrix translate(float x, float y, float z) {
            return Matrix(
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    x, y, z, 1
            );
        }

        /*
         * x 轴旋转矩阵
         */
        static Matrix rotationX(float angle) {
            float s = sin(angle);
            float c = cos(angle);
            return Matrix(
                    1, 0, 0, 0,
                    0, c, s, 0,
                    0, -s, c, 0,
                    0, 0, 0, 1
            );
        }

        /*
         * y 轴旋转矩阵
         */
        static Matrix rotationY(float angle) {
            float s = sin(angle);
            float c = cos(angle);
            return Matrix(
                    c, 0, -s, 0,
                    0, 1, 0, 0,
                    s, 0, c, 0,
                    0, 0, 0, 1
            );
        }

        /*
         * z 轴旋转矩阵
         */
        static Matrix rotationZ(float angle) {
            float s = sin(angle);
            float c = cos(angle);
            return Matrix(
                    c, s, 0, 0,
                    -s, c, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1
            );
        }

        /*
         * 矩阵转置
         */
        static Matrix transpose(const Matrix &mat) {
            float tm[16];
            tm[0] = mat._m[0];
            tm[1] = mat._m[4];
            tm[2] = mat._m[8];
            tm[3] = mat._m[12];
            tm[4] = mat._m[1];
            tm[5] = mat._m[5];
            tm[6] = mat._m[9];
            tm[7] = mat._m[13];
            tm[8] = mat._m[2];
            tm[9] = mat._m[6];
            tm[10] = mat._m[10];
            tm[11] = mat._m[14];
            tm[12] = mat._m[3];
            tm[13] = mat._m[7];
            tm[14] = mat._m[11];
            tm[15] = mat._m[15];
            return Matrix(
                    tm[0], tm[1], tm[2], tm[3],
                    tm[4], tm[5], tm[6], tm[7],
                    tm[8], tm[9], tm[10], tm[11],
                    tm[12], tm[13], tm[14], tm[15]
            );
        };

        /*
+         * 观察矩阵 LookAt
+         */
        static Matrix lookAtLH(const Vector3 &eye, const Vector3 &target, const Vector3 &up) {
            Vector3 z = target - eye;
            z.normalize();
            Vector3 x = Vector3::cross(up, z);
            x.normalize();
            Vector3 y = Vector3::cross(z, x);

            float ex = Vector3::dot(x, eye);
            float ey = Vector3::dot(y, eye);
            float ez = Vector3::dot(z, eye);

            return Matrix(
                    x._x, y._x, z._x, 0,
                    x._y, y._y, z._y, 0,
                    x._z, y._z, z._z, 0,
                    ex, ey, ez, 1);
        }

        /*
         * 透视投影矩阵
         */
        static Matrix perspectiveFovLH(float fov, float aspect, float near, float far) {
            float t = static_cast<float>(1 / (tan(fov * 0.5)));
            return Matrix(
                    t / aspect, 0, 0, 0,
                    0, t, 0, 0,
                    0, 0, -far / (near / far), 1,
                    0, 0, near * far / (near - far), 0
            );
        }

        /*
         * 正交投影矩阵
         */
        static Matrix orthoLH(float width, float height, float near, float far) {
            return Matrix(
                    2 / width, 0, 0, 0,
                    0, 2 / height, 0, 0,
                    0, 0, 1 / (far - near), (-near) / (far - near),
                    0, 0, 0, 1
            );
        }

        /**
         * 变换二维向量
         */
        static Vector2 transfrom(const Vector2 &v, const Matrix &transMat) {
            float x = v._x * transMat._m[0] + v._x * transMat._m[4];
            float y = v._y * transMat._m[1] + v._y * transMat._m[5];
            return Vector2(x, y);
        }

        /*
        * 变换三维向量
        */
        static Vector3 transform(const Vector3 &v3, const Matrix &transMat) {
            float x = v3._x * transMat._m[0] + v3._y * transMat._m[4] + v3._z * transMat._m[8];
            float y = v3._x * transMat._m[1] + v3._y * transMat._m[5] + v3._z * transMat._m[9];
            float z = v3._x * transMat._m[2] + v3._y * transMat._m[6] + v3._z * transMat._m[10];
            return Vector3(x, y, z);
        }

        /*
         * 齐次变换三维向量
         */
        static Vector3 transformCoordinates(const Vector3 &v3, const Matrix &transMat) {
            float x = v3._x * transMat._m[0] + v3._y * transMat._m[4] + v3._z * transMat._m[8] + transMat._m[12];
            float y = v3._x * transMat._m[1] + v3._y * transMat._m[5] + v3._z * transMat._m[9] + transMat._m[13];
            float z = v3._x * transMat._m[2] + v3._y * transMat._m[6] + v3._z * transMat._m[10] + transMat._m[14];
            float w = v3._x * transMat._m[3] + v3._y * transMat._m[7] + v3._z * transMat._m[11] + transMat._m[15];
            return Vector3(x / w, y / w, z / w);
        }


    };
}

#endif //MATHLIB_MATRIX_HPP
