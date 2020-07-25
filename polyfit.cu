#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#define M 12

double* polyfit(double* x, double* y, int n, int M)
{
	int m;
	m = n + 1;
	double **a = (double **)malloc(sizeof(double*)*m);
	for (int i = 0; i < m; i++)
	{
		a[i] = (double*)malloc(m * sizeof(double));
	}
	double *p = (double*)malloc(m * sizeof(double));
	double *b = (double*)malloc(m * sizeof(double));
	double *atemp = (double*)malloc(2 * m * sizeof(double));
	for (int i = 0; i < m; i++)
	{
		b[i] = 0;
		atemp[2 * i] = 0;
		atemp[2 * i + 1] = 0;
	}

	//	构建线性方程组系数矩阵，b[]不变

	for (int i = 0; i < M; i++)
	{
		for (int k = 1; k <= n * 2; k++)
		{
			atemp[k] += pow(x[i], k);
		}
		for (int h = 0; h < n + 1; h++)
		{
			b[h] += pow(x[i], h)*y[i];
		}
	}
	atemp[0] = M;

	for (int i = 0; i < m; i++)
	{
		int k = i;
		for (int g = 0; g < m; g++)
		{
			a[i][g] = atemp[k++];
		}
	}

	//解方程 a*p = b；
	//变上三角	
	for (int i = 0; i < n; i++)
	{
		if (a[i][i] == 0)
		{
			double temp = a[i][i];
			int idx = i;
			while (temp == 0)
			{
				temp = a[idx + 1][i];
				idx = idx + 1;
			}
			//交换第i行和第idx行
			double change, change_b;
			for (int g = 0; g < m; g++)
			{
				change = a[i][g];
				a[i][g] = a[idx][g];
				a[idx][g] = change;
			}
			change_b = b[i];
			b[i] = b[idx];
			b[idx] = change_b;
		}

		if (a[i][i] != 1)
		{
			double temp = a[i][i];
			for (int g = 0; g < m; g++)
			{
				a[i][g] = a[i][g] / temp;
			}
			b[i] = b[i] / temp;
		}

		for (int k = i + 1; k < m; k++)
		{
			if (a[k][i] != 0)
			{
				double temp = -a[k][i];
				for (int p = 0; p < m; p++)
				{
					a[k][p] = a[k][p] + a[i][p] * temp;
				}
				b[k] = b[k] + b[i] * temp;
			}
		}
	}

	if (a[n][n] != 1)
	{
		double temp = a[n][n];
		a[n][n] = 1;
		b[n] = b[n] / temp;
	}

	//消元
	for (int i = n; i > 0; i--)
	{
		for (int g = 0; g < i; g++)
		{
			double temp = -a[g][i];
			b[g] = b[g] + b[i] * temp;
		}
	}

	for (int i = 0; i < m; i++)
	{
		p[i] = b[n - i];
	}

	free(b);
	free(atemp);
	for (int i = 0; i < m; i++)
		free(a[i]);/*释放列*/

	free(a);/*释放行*/
	return p;
	free(p);

}