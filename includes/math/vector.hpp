/*
 * File   : vector
 * Author : KsGin 
 * Date   : 2018/5/23
 */

#ifndef MATHLIB_VECTOR_HPP
#define MATHLIB_VECTOR_HPP

#include <cmath>

namespace Math {

    /**
     *  Vector2 向量类
     */
    class Vector2 {

    public:
        /**
         * 数值定义
         */
        float _x, _y;

        /**
         * 空构造方法
         */
        Vector2() {}

        /**
         * 值构造方法
         */
        Vector2(const float vx, const float vy) {
            _x = vx;
            _y = vy;
        }

        /**
         * 对象拷贝方法
         */
        Vector2(const Vector2 &v2) {
            _x = v2._x;
            _y = v2._y;
        }

        /**
         * 重载 =
         */
        Vector2 &operator=(Vector2 const &v2) = default;

        /**
         * 重载 +
         */
        Vector2 operator+(const Vector2 &v2) const {
            return Vector2(_x + v2._x, _y + v2._y);
        }

        /**
         * 重载 -
         */
        Vector2 operator-(const Vector2 &v2) const {
            return Vector2(_x - v2._x, _y - v2._y);
        }

        /*
         * 重载 *
         */
        Vector2 operator*(const float val) const {
            return Vector2(_x * val , _y * val);
        }

        /**
         * 判断相等
         */
        bool operator==(const Vector2 &v2) const {
            return _x == v2._x && _y == v2._y;
        }

        /**
         * 取负
         */
        Vector2 &negate() {
            _x = -_x;
            _y = -_y;
            return *this;
        }

        /**
         * 缩放
         */
        Vector2 &scale(float scaleX, float scaleY) {
            _x *= scaleX;
            _y *= scaleY;
            return *this;
        }

        /**
         * 求长度
         */
        float length() {
            return sqrt(_x * _x + _y * _y);
        }

        /**
         * 归一化
         */
        Vector2 &normalize() {
            auto length = this->length();
            if (length == 0) return *this;
            _x /= length;
            _y /= length;
            return *this;
        }

    };

    /**
     * Vector3 类
     */
    class Vector3 {
    public:
        float _x, _y, _z;

        /**
         * 空构造方法
         */
        Vector3() {}

        /**
         * 值构造方法
         */
        Vector3(float vx, float vy, float vz) {
            _x = vx;
            _y = vy;
            _z = vz;
        }

        /**
         * 拷贝构造
         */
        Vector3(const Vector3 &v3) {
            _x = v3._x;
            _y = v3._y;
            _z = v3._z;
        }

        /**
         * 重载 =
         */
        Vector3 &operator=(Vector3 const &v3) = default;

        /**
         * 重载 +
         */
        Vector3 operator+(const Vector3 &v3) const {
            return Vector3(_x + v3._x, _y + v3._y, _z + v3._z);
        }

        /**
         * 重载 -
         */
        Vector3 operator-(const Vector3 &v3) const {
            return Vector3(_x - v3._x, _y - v3._y, _z - v3._z);
        }

        /*
         * 重载 *
         */
        Vector3 operator*(const float val) const {
            return Vector3(_x * val , _y * val , _z * val);
        }

        /**
         * 重载 ==
         */
        bool operator==(const Vector3 &v3) const {
            return _x == v3._x && _y == v3._y && _z == v3._z;
        }

        /**
         * 取负
         */
        Vector3 &negate() {
            _x = -_x;
            _y = -_y;
            _z = -_z;
            return *this;
        }

        /**
         * 缩放
         */
        Vector3 &scale(float scaleX, float scaleY, float scaleZ) {
            _x *= scaleX;
            _y *= scaleY;
            _z *= scaleZ;
            return *this;
        }

        /**
         * 求长度
         */
        float length() const {
            return sqrt(_x * _x + _y * _y + _z * _z);
        }

        /**
         * 归一化
         */
        Vector3 &normalize() {
            auto length = this->length();
            if (length == 0) return *this;
            _x /= length;
            _y /= length;
            _z /= length;
            return *this;
        }

        /*
         * 求点积
         */
        static float dot(const Vector3 &vl, const Vector3 &vr) {
            return vl._x * vr._x + vl._y * vr._y + vl._z * vr._z;
        }

        /*
         * 求叉积
         */
        static Vector3 cross(const Vector3 &vl, const Vector3 &vr) {
            float x = vl._y * vr._z - vl._z * vr._y;
            float y = vl._z * vr._x - vl._x * vr._z;
            float z = vl._x * vr._y - vl._y * vr._x;
            return Vector3(x, y, z);
        }
    };
}

#endif //MATHLIB_VECTOR_HPP
