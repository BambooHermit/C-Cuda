#pragma once
#ifndef __FFT_H   //���û�ж���__TEST_H����ô������ 
#define __FFT_H

void fft(cufftDoubleComplex *in, cufftDoubleComplex *out, int size);        //test.c���д��test()���� 
void ifft(cufftDoubleComplex *in, cufftDoubleComplex *out, int size);
void fftshift(cufftDoubleComplex *in, cufftDoubleComplex *out, int size);
void ifftshift(cufftDoubleComplex *in, cufftDoubleComplex *out, int size);

#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"