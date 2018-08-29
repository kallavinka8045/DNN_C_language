#ifndef __MEMALLOC
#define __MEMALLOC
#include <stdlib.h>

double **MemAlloc_D_2D(const int height, const int width); // (double형)2차원 배열을 동적 메모리 할당하는 함수
void MemFree_D_2D(double **arr, const int height); // (double형)2차원 동적 메모리 할당을 해제하는 함수

#endif