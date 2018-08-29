#ifndef __MNIST
#define __MNIST

#include <stdio.h>

#define ROW 28
#define COL 28
#define DATA_LEN ROW*COL
#define ANS_LEN 10
#define IMAGE_OFFSET 16
#define LABEL_OFFSET 8
#define TRAIN_DATA 55000
#define VALID_DATA 5000
#define TEST_DATA 10000
#define BYTE_MAX 256

void Read_MNIST(
	double*** training_data,
	double*** training_answer,
	double*** validation_data,
	double*** validation_answer,
	double*** test_data,
	double*** test_answer
	);

void Free_MNIST(
	double*** training_data,
	double*** training_answer,
	double*** validation_data,
	double*** validation_answer,
	double*** test_data,
	double*** test_answer
	);

#endif