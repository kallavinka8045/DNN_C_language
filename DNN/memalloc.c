#include "memalloc.h"

double **MemAlloc_D_2D(const int height, const int width) // (double형)2차원 배열을 동적 메모리 할당하는 함수
{
	double **arr; //함수 내에서 사용할 2차원 배열을 만들기 위해 2차 포인터 선언
	int i;
	arr = (double**)malloc(sizeof(double*)*height); //높이만큼의 크기로 동적 할당
	for (i = 0; i<height; i++)
	{
		arr[i] = (double*)malloc(sizeof(double)*width); //높이 하나하나마다 너비만큼의 크기로 동적 할당
	}
	return arr; //생성된 2차원 배열 반환
}
void MemFree_D_2D(double **arr, const int height) // (double형)2차원 동적 메모리 할당을 해제하는 함수
{
	int i;
	for (i = 0; i<height; i++)
	{
		free(arr[i]);
	}
	free(arr);
}