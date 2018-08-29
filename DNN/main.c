#include "network.h"
#include "mnist.h"
#include "mathfunc.h"
#define NUM_LAYERS 3

int main()
{
	double **training_data = NULL, **training_answer = NULL, **validation_data = NULL, **validation_answer = NULL, **test_data = NULL, **test_answer = NULL;
	Read_MNIST(&training_data, &training_answer, &validation_data, &validation_answer, &test_data, &test_answer);

	Network net;
	int sizes[NUM_LAYERS] = { 784, 80, 10 };
	double select_dropout_rate[NUM_LAYERS - 2] = { 0.8 };

	// QUADRATIC_COST
	// CROSS_ENTROPY
	int select_cost_function = CROSS_ENTROPY;

	// SIGMOID
	// RELU
	// SOFTMAX
	int select_activate_function[NUM_LAYERS - 1] = { SIGMOID, SOFTMAX };

	Netinit(&net, NUM_LAYERS, sizes, select_cost_function, select_activate_function, select_dropout_rate);

	SGD(
		&net,						 //Network Structure
		training_data,				 //Training data
		training_answer,			 //Training answer data
		TRAIN_DATA,					 //len(Training data)
		validation_data,			 //Training data
		validation_answer,			 //Training answer data
		VALID_DATA,					 //len(Training data)
		test_data,					 //Test data
		test_answer,				 //Test answer data
		TEST_DATA,					 //len(Test data)
		20,							 //Num of epochs
		10,							 //Mini-batch size
		0.01,						 //Learning rate(eta)
		0,							 //L2 Regulariation rate(lambda)
		-1							 //early-stopping patience, if don't use, type -1
		);

	Netdest(&net);
	Free_MNIST(&training_data, &training_answer, &validation_data, &validation_answer, &test_data, &test_answer);
	return 0;
}