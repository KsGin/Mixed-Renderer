/**
 * File Name : tracer.h
 * Author : Yang Fan
 * Date : 2019/5/5
 * declare tracing class
 */

#pragma once

class Tracer {
	

public:
	static Tracer& getInstance() {
		static Tracer tracer;
		return tracer;
	}
};