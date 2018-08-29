#include "mnist.h"
#include "memalloc.h"

void Read_MNIST(
	double*** training_data,
	double*** training_answer,
	double*** validation_data,
	double*** validation_answer,
	double*** test_data,
	double*** test_answer
	)
{
	FILE *fi1, *fi2, *fi3, *fi4;
	unsigned char data[DATA_LEN];
	unsigned char answer;
	int i, j;

	(*training_data) = MemAlloc_D_2D(TRAIN_DATA, DATA_LEN);
	(*training_answer) = MemAlloc_D_2D(TRAIN_DATA, ANS_LEN);
	(*validation_data) = MemAlloc_D_2D(VALID_DATA, DATA_LEN);
	(*validation_answer) = MemAlloc_D_2D(VALID_DATA, ANS_LEN);
	(*test_data) = MemAlloc_D_2D(TEST_DATA, DATA_LEN);
	(*test_answer) = MemAlloc_D_2D(TEST_DATA, ANS_LEN);

	fopen_s(&fi1, "train-images.idx3-ubyte", "rb");
	fopen_s(&fi2, "train-labels.idx1-ubyte", "rb");
	fseek(fi1, sizeof(unsigned char) * IMAGE_OFFSET, SEEK_SET);
	fseek(fi2, sizeof(unsigned char) * LABEL_OFFSET, SEEK_SET);
	for (i = 0; i < TRAIN_DATA; i++)
	{
		fread(data, sizeof(unsigned char), DATA_LEN, fi1);
		fread(&answer, sizeof(unsigned char), 1, fi2);
		{
			for (j = 0; j < DATA_LEN; j++)
				(*training_data)[i][j] = (double)data[j] / BYTE_MAX;
			for (j = 0; j < ANS_LEN; j++)
			{
				if (answer == j) (*training_answer)[i][j] = 1.0;
				else (*training_answer)[i][j] = 0.0;
			}
		}
	}

	for (i = 0; i < VALID_DATA; i++)
	{
		fread(data, sizeof(unsigned char), DATA_LEN, fi1);
		fread(&answer, sizeof(unsigned char), 1, fi2);
		{
			for (j = 0; j < DATA_LEN; j++)
				(*validation_data)[i][j] = (double)data[j] / BYTE_MAX;
			for (j = 0; j < ANS_LEN; j++)
			{
				if (answer == j) (*validation_answer)[i][j] = 1.0;
				else (*validation_answer)[i][j] = 0.0;
			}
		}
	}
	fclose(fi1);
	fclose(fi2);

	fopen_s(&fi3, "t10k-images.idx3-ubyte", "rb");
	fopen_s(&fi4, "t10k-labels.idx1-ubyte", "rb");
	fseek(fi3, sizeof(unsigned char) * IMAGE_OFFSET, SEEK_SET);
	fseek(fi4, sizeof(unsigned char) * LABEL_OFFSET, SEEK_SET);
	for (i = 0; i < TEST_DATA; i++)
	{
		fread(data, sizeof(unsigned char), DATA_LEN, fi3);
		fread(&answer, sizeof(unsigned char), 1, fi4);
		{
			for (j = 0; j < DATA_LEN; j++)
				(*test_data)[i][j] = (double)data[j] / BYTE_MAX;
			for (j = 0; j < ANS_LEN; j++)
			{
				if (answer == j) (*test_answer)[i][j] = 1.0;
				else (*test_answer)[i][j] = 0.0;
			}
		}
	}
	fclose(fi3);
	fclose(fi4);
}

void Free_MNIST(
	double*** training_data,
	double*** training_answer,
	double*** validation_data,
	double*** validation_answer,
	double*** test_data,
	double*** test_answer
	)
{
	MemFree_D_2D((*training_data), TRAIN_DATA);
	MemFree_D_2D((*training_answer), TRAIN_DATA);
	MemFree_D_2D((*validation_data), VALID_DATA);
	MemFree_D_2D((*validation_answer), VALID_DATA);
	MemFree_D_2D((*test_data), TEST_DATA);
	MemFree_D_2D((*test_answer), TEST_DATA);
}