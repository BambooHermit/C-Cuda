#pragma once
#ifndef __FFT_H   //如果没有定义__TEST_H，那么定义它 
#define __FFT_H

void fft(cufftDoubleComplex *in, cufftDoubleComplex *out, int size);        //test.c里编写的test()函数 
void ifft(cufftDoubleComplex *in, cufftDoubleComplex *out, int size);
void fftshift(cufftDoubleComplex *in, cufftDoubleComplex *out, int size);
void ifftshift(cufftDoubleComplex *in, cufftDoubleComplex *out, int size);

#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"