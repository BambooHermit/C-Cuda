#pragma once
#ifndef __POLYFIT_H   //���û�ж���__TEST_H����ô������ 
#define __POLYFIT_H

double* polyfit(double *x, double *y, int n, int M);        //test.c���д��test()���� 

#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"