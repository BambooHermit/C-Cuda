#pragma once
#ifndef __POLYFIT_H   //如果没有定义__TEST_H，那么定义它 
#define __POLYFIT_H

double* polyfit(double *x, double *y, int n, int M);        //test.c里编写的test()函数 

#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"