#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"
#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <complex>

#define BATCH 1

void fft(cufftDoubleComplex *in, cufftDoubleComplex *out, int size)
{

	cufftDoubleComplex *inDev;
	cufftDoubleComplex *outDev;
	cudaMalloc((void **)&inDev, sizeof(cufftDoubleComplex)*size);
	cudaMalloc((void **)&outDev, sizeof(cufftDoubleComplex)*size);
	cufftHandle plan;
	cufftPlan1d(&plan, size, CUFFT_Z2Z, BATCH);

	cudaMemcpy(inDev, in, sizeof(cufftDoubleComplex)*size, cudaMemcpyHostToDevice);
	cufftExecZ2Z(plan, inDev, outDev, -1);
	cudaDeviceSynchronize();
	cudaMemcpy(out, outDev, sizeof(cufftDoubleComplex)*size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cufftDestroy(plan);
	cudaFree(inDev);
	cudaFree(outDev);
}


void ifft(cufftDoubleComplex *in, cufftDoubleComplex *out, int size)
{

	cufftDoubleComplex *inDev;
	cufftDoubleComplex *outDev;
	cudaMalloc((void **)&inDev, sizeof(cufftDoubleComplex)*size);
	cudaMalloc((void **)&outDev, sizeof(cufftDoubleComplex)*size);
	cufftDoubleComplex N;
	N.x = (double)size;
	N.y = (double)size;
	cufftHandle plan;
	cufftPlan1d(&plan, size, CUFFT_Z2Z, BATCH);

	cudaMemcpy(inDev, in, sizeof(cufftDoubleComplex)*size, cudaMemcpyHostToDevice);
	cufftExecZ2Z(plan, inDev, outDev, 1);
	cudaDeviceSynchronize();
	cudaMemcpy(out, outDev, sizeof(cufftDoubleComplex)*size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	for (int i = 0; i < size; i++)
	{
		out[i].x = out[i].x / N.x;
		out[i].y = out[i].y / N.y;
	}

	cufftDestroy(plan);
	cudaFree(inDev);
	cudaFree(outDev);
}

void fftshift(cufftDoubleComplex *in, cufftDoubleComplex *out, int size)
{
	int idx = size / 2;

	for (int i = 0; i < idx; i++)
	{
		out[i] = in[size - idx + i];
		out[idx + i] = in[i];
	}

	if ((size % 2) == 1)
	{
		out[size - 1] = in[idx];
	}
}

void ifftshift(cufftDoubleComplex *in, cufftDoubleComplex *out, int size)
{
	int idx = size / 2;

	for (int i = 0; i < idx; i++)
	{
		out[size - idx + i] = in[i];
		out[i] = in[idx + i];
	}

	if ((size % 2) == 1)
	{
		out[idx] = in[size - 1];
	}

}