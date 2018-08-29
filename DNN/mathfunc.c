#include "mathfunc.h"

double dot_1d(double* x, double* y, int len)
{
	double sum = 0;
	int i;
	for (i = 0; i < len; i++)
		sum += x[i] * y[i];
	return sum;
}

void transpose(double **x, int size)
{
	int i, j;
	for (i = 0; i<size; i++)
	{
		for (j = i + 1; j<size; j++)
		{
			double temp = x[i][j];
			x[i][j] = x[j][i];
			x[j][i] = temp;
		}
	}
}

double Gaussian(double mean, double std_dev)
{
	double rand1, rand2;
	const double PI = 3.14159265358979323846;
	double norm = 0;
	for (;;)
	{
		rand1 = (double)rand() / RAND_MAX;
		rand2 = (double)rand() / RAND_MAX;
		if (rand1 != 0 && rand2 != 0) break;
	}
	norm = sqrt(-2 * log(rand1))*cos(2 * PI*rand2); //Box-Muller ¹æ¹ý
	return (norm * std_dev + mean);
}

void shuffle_2d(double **array, double **ans, int len_1d, int len_array, int len_ans)
{
	double *tmp = (double*)malloc(sizeof(double)*len_1d);
	int stride1 = len_array*sizeof(double);
	int stride2 = len_ans*sizeof(double);
	int i;
	for (i = 0; i < 2 * len_1d; i++)
	{
		int Dst = rand() % len_1d;
		int Src = rand() % len_1d;

		memcpy(tmp, array[Dst], stride1);
		memcpy(array[Dst], array[Src], stride1);
		memcpy(array[Src], tmp, stride1);

		memcpy(tmp, ans[Dst], stride2);
		memcpy(ans[Dst], ans[Src], stride2);
		memcpy(ans[Src], tmp, stride2);
	}
	free(tmp);
}
