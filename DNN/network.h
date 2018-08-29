#ifndef __NETWORK
#define __NETWORK

#include <stdio.h>
#include <stdlib.h>

#define ON 1
#define OFF 0

#define QUADRATIC_COST 0
#define CROSS_ENTROPY 1
#define SIGMOID 0
#define RELU 1
#define TANH 2
#define SOFTMAX 3

typedef struct
{
	int num_layers;
	int* sizes;
	double** biases;
	double*** weights;
	int** dropout_layers;
	double* select_dropout_rate;
	int select_cost_function;
	int* select_activate_function;
}Network;

void Netinit(Network* net, int num_layers, int* sizes, int select_cost_function, int* select_activate_function, double* select_dropout_rate);
void Netdest(Network* net);

void SGD(
	Network* net,
	double** training_data,
	double** training_answer,
	int len_train,
	double** validation_data,
	double** validation_answer,
	int len_valid,
	double** test_data,
	double** test_answer,
	int len_test,
	int epochs,
	int mini_batch_size,
	double learning_rate,
	double regularization_rate,
	int max_patience
	);
void update_mini_batch(
	Network* net,
	double** mini_batch,
	double** mini_batch_answer,
	int mini_batch_size,
	double learning_rate,
	double regularization_rate,
	int len_train
	);
void backprop(
	Network* net,
	double* mini_batch,
	double* mini_batch_answer,
	double** delta_nabla_b,
	double*** delta_nabla_w
	);
void dropout(Network* net);

int evaluate(Network* net, double** test_data, double** test_answer, int len_test);
void feedforward(Network* net, double* input, double* z_for_softmax, double* output);

double QuadraticCost(Network* net, double** y, double** a, int n, double regularization_rate);
double CrossEntropyCost(Network* net, double** y, double** z_outlayer, double** a, int n, double regularization_rate);
void cost_derivate(Network* net, double* output_activated, double* y, double* delta);
double sigmoid(double z);
double sigmoid_prime(double z);
double ReLU(double z);
double ReLU_prime(double z);
double tanh_activate(double z);
double tanh_prime(double z);
void softmax(Network *net, double** a, double** z, int i);
double log_softmax(Network *net, double z, double* z_outlayer);
double softmax_prime(double a);
#endif