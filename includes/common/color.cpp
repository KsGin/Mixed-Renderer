/*
 * File   : color
 * Author : KsGin 
 * Date   : 2018/6/3
 */

#include "color.h"

Color::Color() {
    this->x = 0;
    this->y = 0;
    this->z = 0;
    this->w = 0;
}

Color::Color(float r, float g, float b, float a) {
    this->x = r;
    this->y = g;
    this->z = b;
    this->w = a;
}

Color::Color(const Color &color) {
    this->r = color.r;
    this->g = color.g;
    this->b = color.b;
    this->a = color.a;
}

Color &Color::operator=(const Color &color) {
    this->r = color.r;
    this->g = color.g;
    this->b = color.b;
    this->a = color.a;
    return *this;
}

Color Color::operator+(const Color &color) {
    return Color(r + color.r, g + color.g, b + color.b, a + color.a);
}

Color Color::operator-(const Color &color) {
    return Color(r - color.r, g - color.g, b - color.b, a - color.a);
}

Color Color::operator*(const Color &color) {
    return Color(r * color.r, g * color.g, b * color.b, a * color.a);
}

Color Color::operator/(const Color &color) {
    return Color(r / color.r, g / color.g, b / color.b, a / color.a);
}

Color Color::operator+(const float c) {
    return Color(r + c, g + c, b + c, a + c);
}

Color Color::operator-(const float c) {
    return Color(r - c, g - c, b - c, a - c);
}

Color Color::operator*(const float c) {
    return Color(r * c, g * c, b * c, a * c);
}

Color Color::operator/(const float c) {
    return Color(r / c, g / c, b / c, a / c);
}

Color Color::red() {
    return Color(1, 0, 0, 1);
}

Color Color::green() {
    return Color(0, 1, 0, 1);
}

Color Color::blue() {
    return Color(0, 0, 1, 0);
}

Color Color::white() {
    return Color(1, 1, 1, 1);
}

Color Color::black() {
    return Color(0, 0, 0, 0);
}

Color &Color::modulate() {
    this->r = (r > 0 ? r : 0) < 1 ? r : 1;
    this->g = (g > 0 ? g : 0) < 1 ? g : 1;
    this->b = (b > 0 ? b : 0) < 1 ? b : 1;
    this->a = (a > 0 ? a : 0) < 1 ? a : 1;
    return *this;
}
