#ifndef __MATHFUNC
#define __MATHFUNC
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

double dot_1d(double* x, double* y, int len);
void MatTranspose(double **x, const int size);
double Gaussian(double mean, double std_dev);
void shuffle_2d(double **array, double **ans, int len_1d, int len_array, int len_ans);

#endif