/**
 * File Name : shader.h
 * Author : Yang Fan
 * Date : 2018/12/19
 * declare pipe shader
 */
#pragma once

#include "../includes/math/matrix.hpp"
#include "../includes/math/vector.hpp"

static class Shader
{
	static void transform(const Math::Matrix &mat) {

	}

	static void vs() {

	}

	static void ps() {

	}

public:
	static void render(const Math::Matrix &mat) {
		transform(mat);

		vs();

		ps();
	}
};
