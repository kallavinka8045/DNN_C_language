#include "network.h"
#include "mathfunc.h"

void Netinit(Network* net, int num_layers, int* sizes, int select_cost_function, int* select_activate_function, double* select_dropout_rate)
{
	int i, j, k;

	//(1) layer 갯수 초기화
	net->num_layers = num_layers;

	//(2) layer 크기 초기화
	net->sizes = (int*)malloc(sizeof(int)*num_layers);
	for (i = 0; i < num_layers; i++)
		net->sizes[i] = sizes[i];

	//(3) 각 neuron의 bias 초기화
	net->biases = (double**)malloc(sizeof(double*)*(num_layers-1));
	for (i = 0; i < num_layers-1; i++)
		net->biases[i] = (double*)malloc(sizeof(double)*sizes[i+1]);
	for (i = 0; i < num_layers-1; i++)
		for (j = 0; j < sizes[i+1]; j++)
			net->biases[i][j] = Gaussian(0, 1 / sqrt(sizes[0]));

	//(4) 각 neuron의 weight 초기화
	net->weights = (double***)malloc(sizeof(double**)*(num_layers-1));
	for (i = 0; i < num_layers-1; i++)
	{
		net->weights[i] = (double**)malloc(sizeof(double*)*sizes[i+1]);
		for (j = 0; j < sizes[i+1]; j++)
			net->weights[i][j] = (double*)malloc(sizeof(double)*sizes[i]);
	}
	for (i = 0; i < num_layers; i++)
		for (j = 0; j < sizes[i+1]; j++)
			for (k = 0; k < sizes[i]; k++)
				net->weights[i][j][k] = Gaussian(0, 1 / sqrt(sizes[0]));

	//(3) 각 neuron의 dropout 여부를 나타내는 배열 초기화
	net->select_dropout_rate = (double*)malloc(sizeof(double)*(num_layers - 2));
	for (i = 0; i < num_layers - 2; i++)
		net->select_dropout_rate[i] = select_dropout_rate[i];
	net->dropout_layers = (int**)malloc(sizeof(int*)*(num_layers - 2));
	for (i = 0; i < num_layers - 2; i++)
		net->dropout_layers[i] = (int*)malloc(sizeof(int)*sizes[i + 1]);
	for (i = 0; i < num_layers - 2; i++)
		for (j = 0; j < sizes[i + 1]; j++)
			net->dropout_layers[i][j] = ON;


	//(6)어떤 cost function을 선택했는지 저장
	if (select_cost_function == QUADRATIC_COST) net->select_cost_function = QUADRATIC_COST;
	else if (select_cost_function == CROSS_ENTROPY) net->select_cost_function = CROSS_ENTROPY;
	else
	{
		printf("Select Cost Function ERROR\n");
		exit(0);
	}

	//(7)각 출력단마다 어떤 activation function을 선택했는지 저장
	net->select_activate_function = (int*)malloc(sizeof(int)*(num_layers - 1));

	for (i = 0; i < num_layers - 1; i++)
	{
		if (select_activate_function[i] == SIGMOID) net->select_activate_function[i] = SIGMOID;
		else if (select_activate_function[i] == RELU) net->select_activate_function[i] = RELU;
		else if (select_activate_function[i] == TANH) net->select_activate_function[i] = TANH;
		else if (select_activate_function[i] == SOFTMAX) net->select_activate_function[i] = SOFTMAX;
		else
		{
			printf("Select Activation Function ERROR\n");
			exit(0);
		}
	}

	char* string_costfunc = NULL;
	char** string_activatefunc = (char**)malloc(sizeof(char*)*num_layers-1);

	time_t timer = time(NULL);
	struct tm t;
	localtime_s(&t, &timer);

	printf("실행 시각 : %d년 %d월 %d일 %d시 %d분 %d초\n\n", t.tm_year + 1900, t.tm_mon+1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
	printf("Deep Neural Network (Using CPU)\n");
	if (select_cost_function == QUADRATIC_COST)
	{
		string_costfunc = "Quadratic";
		printf("[%s] Cost function is selected\n", string_costfunc);
	}
	else if (select_cost_function == CROSS_ENTROPY)
	{
		string_costfunc = "Categorical Cross-Entropy";
		printf("[%s] Cost function is selected\n", string_costfunc);
	}
	for (i = 0; i < num_layers - 1; i++)
	{
		if (select_activate_function[i] == SIGMOID) string_activatefunc[i] = "Sigmoid";
		else if (select_activate_function[i] == RELU) string_activatefunc[i] = "ReLU";
		else if (select_activate_function[i] == TANH) string_activatefunc[i] = "tanh";
		else if (select_activate_function[i] == SOFTMAX) string_activatefunc[i] = "Softmax";

	}

	printf("---------------------------------------------------------------------------\n");
	printf("Layer		Shape	Param	Dropout	Actfunc	   \n");
	printf("===========================================================================\n");

	for (i = 0; i < num_layers; i++)
	{
		if (i == 0)
		{
			printf("Input Layer	%-4d	-	-	-\n", net->sizes[i]);
			printf("---------------------------------------------------------------------------\n");
		}
		else if (i == num_layers-1)
		{
			printf("Output Layer	%-4d	%d	-	%s\n", net->sizes[i], net->sizes[i] + net->sizes[i] * net->sizes[i - 1], string_activatefunc[i - 1]);
			printf("---------------------------------------------------------------------------\n");
		}
		else
		{
			printf("Hidden Layer #%d	%-4d	%d	%.4f	%s\n", i, net->sizes[i], net->sizes[i] + net->sizes[i] * net->sizes[i - 1], net->select_dropout_rate[i - 1], string_activatefunc[i - 1]);
			printf("---------------------------------------------------------------------------\n");
		}
	}

	free(string_activatefunc);
}
void Netdest(Network* net)
{
	int num_layers = net->num_layers;
	int i, j;
	free(net->sizes);
	for (i = 0; i < num_layers - 1; i++)
		free(net->biases[i]);
	free(net->biases);
	for (i = 0; i < num_layers - 1; i++)
	{
		for (j = 0; j < net->sizes[i+1]; j++)
			free(net->weights[i][j]);
		free(net->weights[i]);
	}
	free(net->weights);
	for (i = 0; i < num_layers - 2; i++)
		free(net->dropout_layers[i]);
	free(net->dropout_layers);
	free(net->select_dropout_rate);
	free(net->select_activate_function);
}

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
	)
{
	int h, i, j, k;
	int num_layers = net->num_layers;
	int train_result, valid_result, test_result;
	double train_cost, valid_cost, test_cost;
	int check_acc = 0, check_idx = 0, patience = 0;

	double** best_biases = (double**)malloc(sizeof(double*)*(num_layers - 1));
	for (i = 0; i < num_layers - 1; i++)
		best_biases[i] = (double*)malloc(sizeof(double)*net->sizes[i + 1]);
	double*** best_weights = (double***)malloc(sizeof(double**)*(num_layers - 1));
	for (i = 0; i < num_layers - 1; i++)
	{
		best_weights[i] = (double**)malloc(sizeof(double*)*net->sizes[i + 1]);
		for (j = 0; j < net->sizes[i + 1]; j++)
			best_weights[i][j] = (double*)malloc(sizeof(double)*net->sizes[i]);
	}

	//미니 배치 동적 할당
	double** mini_batch = (double**)malloc(sizeof(double*)*mini_batch_size);
	for (i = 0; i < mini_batch_size; i++)
		mini_batch[i] = (double*)malloc(sizeof(double)*net->sizes[0]);
	double** mini_batch_answer = (double**)malloc(sizeof(double*)*mini_batch_size);
	for (i = 0; i < mini_batch_size; i++)
		mini_batch_answer[i] = (double*)malloc(sizeof(double)*net->sizes[num_layers - 1]);

	double** train_activated = (double**)malloc(sizeof(double*)*len_train);
	for (i = 0; i < len_train; i++)
		train_activated[i] = (double*)malloc(sizeof(double)*net->sizes[num_layers - 1]);
	double** validation_activated = (double**)malloc(sizeof(double*)*len_valid);
	for (i = 0; i < len_valid; i++)
		validation_activated[i] = (double*)malloc(sizeof(double)*net->sizes[num_layers - 1]);
	double** test_activated = (double**)malloc(sizeof(double*)*len_test);
	for (i = 0; i < len_test; i++)
		test_activated[i] = (double*)malloc(sizeof(double)*net->sizes[num_layers - 1]);

	double** train_z_for_softmax = (double**)malloc(sizeof(double*)*len_train); //log(softmax) = log(0)으로 인해 underflow 방지용
	for (i = 0; i < len_train; i++)
		train_z_for_softmax[i] = (double*)malloc(sizeof(double)*net->sizes[num_layers - 1]);
	double** valid_z_for_softmax = (double**)malloc(sizeof(double*)*len_valid);
	for (i = 0; i < len_valid; i++)
		valid_z_for_softmax[i] = (double*)malloc(sizeof(double)*net->sizes[num_layers - 1]);
	double** test_z_for_softmax = (double**)malloc(sizeof(double*)*len_test);
	for (i = 0; i < len_test; i++)
		test_z_for_softmax[i] = (double*)malloc(sizeof(double)*net->sizes[num_layers - 1]);

	printf("[Stochastic Gradient Descent] optimizer is selected\n");
	printf("%d mini batch size / %.4f eta / %.4f lambda\n\n", mini_batch_size, learning_rate, regularization_rate);

	//training 단계
	for (h = 0; h < epochs; h++)
	{
		shuffle_2d(training_data, training_answer, len_train, net->sizes[0], net->sizes[num_layers - 1]);
		//mini_batch_size 단위로 묶음
		for (i = 0; i < len_train / mini_batch_size; i++)
		{
			for (j = 0; j < mini_batch_size; j++)
			{
				for (k = 0; k < net->sizes[0]; k++)
					mini_batch[j][k] = training_data[mini_batch_size*i + j][k];
				for (k = 0; k < net->sizes[num_layers - 1]; k++)
					mini_batch_answer[j][k] = training_answer[mini_batch_size*i + j][k];
			}
			update_mini_batch(net, mini_batch, mini_batch_answer, mini_batch_size, learning_rate, regularization_rate, len_train);
			printf("\rEpoch %*d/%d : %*d/%d",
				(int)log10(epochs) + 1, h + 1, epochs, (int)log10(len_train / mini_batch_size) + 2, (i + 1)*mini_batch_size, len_train);
		}
		//training data의 loss와 cost 계산
		for (i = 0; i < len_train; i++)
			feedforward(net, training_data[i], train_z_for_softmax[i], train_activated[i]);
		train_result = evaluate(net, train_activated, training_answer, len_train);
		if (net->select_cost_function == QUADRATIC_COST) train_cost = QuadraticCost(net, training_answer, train_activated, len_train, regularization_rate);
		else if (net->select_cost_function == CROSS_ENTROPY) train_cost = CrossEntropyCost(net, training_answer, train_z_for_softmax, train_activated, len_train, regularization_rate);

		//validation data의 loss와 cost 계산
		for (i = 0; i < len_valid; i++)
			feedforward(net, validation_data[i], valid_z_for_softmax[i], validation_activated[i]);
		valid_result = evaluate(net, validation_activated, validation_answer, len_valid);
		if (net->select_cost_function == QUADRATIC_COST) valid_cost = QuadraticCost(net, validation_answer, validation_activated, len_valid, regularization_rate);
		else if (net->select_cost_function == CROSS_ENTROPY) valid_cost = CrossEntropyCost(net, validation_answer, valid_z_for_softmax, validation_activated, len_valid, regularization_rate);

		printf(" - loss: %.4f - acc: %.4f - val_loss: %.4f - val_acc: %.4f", train_cost, train_result / (double)len_train, valid_cost, valid_result / (double)len_valid);

		//early-stopping check
		if (max_patience <= -1)
		{
			if (check_acc < valid_result)
			{
				patience = 0;
				for (i = 0; i < num_layers; i++)
					for (j = 0; j < net->sizes[i + 1]; j++)
						for (k = 0; k < net->sizes[i]; k++)
							best_weights[i][j][k] = net->weights[i][j][k];
				for (i = 0; i < num_layers; i++)
					for (j = 0; j < net->sizes[i + 1]; j++)
						best_biases[i][j] = net->biases[i][j];
				check_acc = valid_result;
				check_idx = h + 1;
			}
			else patience++;
			printf(" [patience %d]\n", patience);
			if (patience == max_patience)
			{
				printf("Stop learning...\n");
				printf("Parameters are updated to [epoch %d]\n", check_idx);
				for (i = 0; i < num_layers; i++)
					for (j = 0; j < net->sizes[i + 1]; j++)
						for (k = 0; k < net->sizes[i]; k++)
							net->weights[i][j][k] = best_weights[i][j][k];
				for (i = 0; i < num_layers; i++)
					for (j = 0; j < net->sizes[i + 1]; j++)
						net->biases[i][j] = best_biases[i][j];
				break;
			}
		}
	}
	//test data의 loss와 cost 계산
	for (i = 0; i < len_test; i++)
		feedforward(net, test_data[i], test_z_for_softmax[i], test_activated[i]);
	test_result = evaluate(net, test_activated, test_answer, len_test);
	if (net->select_cost_function == QUADRATIC_COST) test_cost = QuadraticCost(net, test_answer, test_activated, len_test, regularization_rate);
	else if (net->select_cost_function == CROSS_ENTROPY) test_cost = CrossEntropyCost(net, test_answer, test_z_for_softmax, test_activated, len_test, regularization_rate);

	printf("\nTest loss : %.4f\n", test_cost);
	printf("Test accuracy : %*d/%d (%.4f%%)\n", (int)log10(len_test), test_result, len_test, test_result*100/(double)len_test);


	//초기화
	for (i = 0; i < num_layers - 1; i++)
		free(best_biases[i]);
	free(best_biases);
	for (i = 0; i < num_layers - 1; i++)
	{
		for (j = 0; j < net->sizes[i + 1]; j++)
			free(best_weights[i][j]);
		free(best_weights[i]);
	}
	free(best_weights);

	for (i = 0; i < mini_batch_size; i++)
		free(mini_batch[i]);
	free(mini_batch);
	for (i = 0; i < mini_batch_size; i++)
		free(mini_batch_answer[i]);
	free(mini_batch_answer);

	for (i = 0; i < len_train; i++)
		free(train_activated[i]);
	free(train_activated);
	for (i = 0; i < len_valid; i++)
		free(validation_activated[i]);
	free(validation_activated);
	for (i = 0; i < len_test; i++)
		free(test_activated[i]);
	free(test_activated);

	for (i = 0; i < len_train; i++)
		free(train_z_for_softmax[i]);
	free(train_z_for_softmax);
	for (i = 0; i < len_valid; i++)
		free(valid_z_for_softmax[i]);
	free(valid_z_for_softmax);
	for (i = 0; i < len_test; i++)
		free(test_z_for_softmax[i]);
	free(test_z_for_softmax);
}

void update_mini_batch(
	Network* net,
	double** mini_batch,
	double** mini_batch_answer,
	int mini_batch_size,
	double learning_rate,
	double regularization_rate, 
	int len_train
	)
{
	int h, i, j, k;

	double** nabla_b, ***nabla_w, **delta_nabla_b, ***delta_nabla_w;

	//nabla_b 동적 할당
	nabla_b = (double**)malloc(sizeof(double*)*(net->num_layers - 1));
	for (i = 0; i < net->num_layers - 1; i++)
		nabla_b[i] = (double*)malloc(sizeof(double)*net->sizes[i + 1]);
	for (i = 0; i < net->num_layers - 1; i++)
		for (j = 0; j < net->sizes[i + 1]; j++)
			nabla_b[i][j] = 0;

	//nabla_w 동적 할당
	nabla_w = (double***)malloc(sizeof(double**)*(net->num_layers - 1));
	for (i = 0; i < net->num_layers - 1; i++)
	{
		nabla_w[i] = (double**)malloc(sizeof(double*)*net->sizes[i+1]);
		for (j = 0; j < net->sizes[i+1]; j++)
			nabla_w[i][j] = (double*)malloc(sizeof(double)*net->sizes[i]);
	}
	for (i = 0; i < net->num_layers-1; i++)
		for (j = 0; j < net->sizes[i+1]; j++)
			for (k = 0; k < net->sizes[i]; k++)
				nabla_w[i][j][k] = 0;

	//delta_nabla_b 동적 할당
	delta_nabla_b = (double**)malloc(sizeof(double*)*(net->num_layers - 1));
	for (i = 0; i < net->num_layers - 1; i++)
		delta_nabla_b[i] = (double*)malloc(sizeof(double)*net->sizes[i + 1]);

	//delta_nabla_w 동적 할당
	delta_nabla_w = (double***)malloc(sizeof(double**)*(net->num_layers - 1));
	for (i = 0; i < net->num_layers - 1; i++)
	{
		delta_nabla_w[i] = (double**)malloc(sizeof(double*)*net->sizes[i+1]);
		for (j = 0; j < net->sizes[i+1]; j++)
			delta_nabla_w[i][j] = (double*)malloc(sizeof(double)*net->sizes[i]);
	}

	//backprop...
	for (h = 0; h < mini_batch_size; h++)
	{
		backprop(net, mini_batch[h], mini_batch_answer[h], delta_nabla_b, delta_nabla_w);
		for (i = 0; i < net->num_layers - 1; i++)
			for (j = 0; j < net->sizes[i + 1]; j++)
				nabla_b[i][j] += delta_nabla_b[i][j];
		for (i = 0; i < net->num_layers - 1; i++)
			for (j = 0; j < net->sizes[i+1]; j++)
				for (k = 0; k < net->sizes[i]; k++)
					nabla_w[i][j][k] += delta_nabla_w[i][j][k];
	}

	//gradient descent update...
	for (i = 0; i < net->num_layers - 1; i++)
		for (j = 0; j < net->sizes[i + 1]; j++)
			net->biases[i][j] = net->biases[i][j] - (learning_rate / mini_batch_size)*nabla_b[i][j];

	for (i = 0; i < net->num_layers - 1; i++)
		for (j = 0; j < net->sizes[i+1]; j++)
			for (k = 0; k < net->sizes[i]; k++)
				net->weights[i][j][k] = (1 - learning_rate*regularization_rate / len_train)*net->weights[i][j][k] - (learning_rate / mini_batch_size)*nabla_w[i][j][k];

	//alloc free
	for (i = 0; i < net->num_layers - 1; i++)
		free(nabla_b[i]);
	free(nabla_b);
	for (i = 0; i < net->num_layers - 1; i++)
	{
		for (j = 0; j < net->sizes[i+1]; j++)
			free(nabla_w[i][j]);
		free(nabla_w[i]);
	}
	free(nabla_w);

	for (i = 0; i < net->num_layers - 1; i++)
		free(delta_nabla_b[i]);
	free(delta_nabla_b);
	for (i = 0; i < net->num_layers - 1; i++)
	{
		for (j = 0; j < net->sizes[i+1]; j++)
			free(delta_nabla_w[i][j]);
		free(delta_nabla_w[i]);
	}
	free(delta_nabla_w);
}

void backprop(
	Network* net,
	double* mini_batch,
	double* mini_batch_answer,
	double** delta_nabla_b,
	double*** delta_nabla_w
	)
{
	int h, i, j, k;
	double **z, **activated, **delta, *weights_temp;
	int num_layers = net->num_layers;

	for (i = 0; i < net->num_layers - 1; i++)
		for (j = 0; j < net->sizes[i + 1]; j++)
			delta_nabla_b[i][j] = 0;
	for (i = 0; i < net->num_layers - 1; i++)
		for (j = 0; j < net->sizes[i + 1]; j++)
			for (k = 0; k < net->sizes[i]; k++)
				delta_nabla_w[i][j][k] = 0;

	//z 동적 할당
	z = (double**)malloc(sizeof(double*)*(net->num_layers - 1));
	for (i = 0; i < net->num_layers - 1; i++)
		z[i] = (double*)malloc(sizeof(double)*net->sizes[i + 1]);

	//activated 동적 할당
	activated = (double**)malloc(sizeof(double*)*(net->num_layers));
	for (i = 0; i < net->num_layers; i++)
		activated[i] = (double*)malloc(sizeof(double)*net->sizes[i]);
	for (j = 0; j < net->sizes[0]; j++)
		activated[0][j] = mini_batch[j];

	//delta 동적 할당
	delta = (double**)malloc(sizeof(double*)*(net->num_layers - 1));
	for (i = 0; i < net->num_layers - 1; i++)
		delta[i] = (double*)malloc(sizeof(double)*net->sizes[i + 1]);

	//dropout - 확률적으로 각 레이어의 뉴런 랜덤 비활성화
	dropout(net);

	//feedforward step
	for (i = 0; i < net->num_layers - 1; i++)
	{
		for (j = 0; j < net->sizes[i + 1]; j++)
		{
			z[i][j] = dot_1d(net->weights[i][j], activated[i], net->sizes[i]) + net->biases[i][j];
			if (i != num_layers - 2) z[i][j] *= net->dropout_layers[i][j]; //if hidden layers
			if (net->select_activate_function[i] == SIGMOID) activated[i + 1][j] = sigmoid(z[i][j]);
			else if (net->select_activate_function[i] == RELU) activated[i + 1][j] = ReLU(z[i][j]);
			else if (net->select_activate_function[i] == TANH) activated[i + 1][j] = tanh_activate(z[i][j]);
		}
		if (net->select_activate_function[i] == SOFTMAX) softmax(net, activated, z, i);
	}

	//backward step
	for (h = num_layers - 1; h >= 1; h--)
	{
		if (h == num_layers - 1)
		{
			//calculate delta(output layer)
			cost_derivate(net, activated[h], mini_batch_answer, delta[h-1]);
			if (net->select_cost_function == QUADRATIC_COST)
			{
				for (i = 0; i < net->sizes[h]; i++)
				{
					if (net->select_activate_function[h - 1] == SIGMOID) delta[h - 1][i] *= sigmoid_prime(z[h - 1][i]);
					else if (net->select_activate_function[h - 1] == RELU) delta[h - 1][i] *= ReLU_prime(z[h - 1][i]);
					else if (net->select_activate_function[h - 1] == TANH) delta[h - 1][i] *= tanh_prime(z[h - 1][i]);
					else if (net->select_activate_function[h - 1] == SOFTMAX) delta[h - 1][i] *= softmax_prime(activated[h][i]);
				}
			}
		}
		else
		{
			//calculate delta(all layers)
			weights_temp = (double*)malloc(sizeof(double)*net->sizes[h + 1]);
			for (i = 0; i < net->sizes[h]; i++)
			{
				for (j = 0; j < net->sizes[h + 1]; j++)
					weights_temp[j] = net->weights[h][j][i];
				delta[h - 1][i] = net->dropout_layers[h - 1][i] * dot_1d(weights_temp, delta[h], net->sizes[h + 1]);
				if (net->select_activate_function[h - 1] == SIGMOID) delta[h - 1][i] *= sigmoid_prime(z[h - 1][i]);
				else if (net->select_activate_function[h - 1] == RELU) delta[h - 1][i] *= ReLU_prime(z[h - 1][i]);
				else if (net->select_activate_function[h - 1] == TANH) delta[h - 1][i] *= tanh_prime(z[h - 1][i]);
				else if (net->select_activate_function[h - 1] == SOFTMAX) delta[h - 1][i] *= softmax_prime(activated[h][i]);
			}
			free(weights_temp);
		}

		for (j = 0; j < net->sizes[h]; j++)
		{
			delta_nabla_b[h - 1][j] = delta[h - 1][j];
			for (k = 0; k < net->sizes[h - 1]; k++)
				delta_nabla_w[h - 1][j][k] = delta[h - 1][j] * activated[h - 1][k];
		}
	}

	//alloc free
	for (i = 0; i < net->num_layers - 1; i++)
		free(z[i]);
	free(z);
	for (i = 0; i < net->num_layers; i++)
		free(activated[i]);
	free(activated);
	for (i = 0; i < net->num_layers - 1; i++)
		free(delta[i]);
	free(delta);
}
void dropout(Network* net)
{
	int i, j;
	int num_layers = net->num_layers;
	for (i = 0; i < num_layers - 2; i++)
	{
		for (j = 0; j < net->sizes[i + 1]; j++)
		{
			if((double)rand() / RAND_MAX <= net->select_dropout_rate[i]) net->dropout_layers[i][j] = ON;
			else net->dropout_layers[i][j] = OFF;
		}
	}

}


int evaluate(Network* net, double** data_activated, double** data_answer, int len_data)
{
	int i, j;
	int num_layers = net->num_layers;
	int argmax_idx = 0, ans_argmax_idx = 0;
	double argmax = 0, ans_argmax = 0;
	int result = 0;

	for (i = 0; i < len_data; i++)
	{
		argmax = 0;
		argmax_idx = 0;
		ans_argmax = 0;
		ans_argmax_idx = 0;
		for (j = 0; j < net->sizes[num_layers - 1]; j++)
		{
			if (argmax <= data_activated[i][j])
			{
				argmax = data_activated[i][j];
				argmax_idx = j;
			}
			if (ans_argmax <= data_answer[i][j])
			{
				ans_argmax = data_answer[i][j];
				ans_argmax_idx = j;
			}
		}
		if (argmax_idx == ans_argmax_idx) result++;
	}
	return result;
}

void feedforward(Network* net, double* input, double* z_for_softmax, double* output)
{
	int i, j;
	double** a, **z;
	int num_layers = net->num_layers;
	//activated 동적 할당
	a = (double**)malloc(sizeof(double*)*num_layers);
	for (i = 0; i < num_layers; i++)
		a[i] = (double*)malloc(sizeof(double)*net->sizes[i]);
	for (j = 0; j < net->sizes[0]; j++)
		a[0][j] = input[j];
	z = (double**)malloc(sizeof(double*)*(num_layers - 1));
	for (i = 0; i < num_layers - 1;i++)
		z[i] = (double*)malloc(sizeof(double)*net->sizes[i+1]);


	for (i = 0; i < num_layers - 1; i++)
	{
		for (j = 0; j < net->sizes[i + 1]; j++)
		{
			if (net->select_activate_function[i] == SIGMOID)
				a[i + 1][j] = sigmoid(dot_1d(net->weights[i][j], a[i], net->sizes[i]) + net->biases[i][j]);
			else if (net->select_activate_function[i] == RELU)
				a[i + 1][j] = ReLU(dot_1d(net->weights[i][j], a[i], net->sizes[i]) + net->biases[i][j]);
			else if (net->select_activate_function[i] == TANH)
				a[i + 1][j] = tanh_activate(dot_1d(net->weights[i][j], a[i], net->sizes[i]) + net->biases[i][j]);
			else if (net->select_activate_function[i] == SOFTMAX)
				z[i][j] = dot_1d(net->weights[i][j], a[i], net->sizes[i]) + net->biases[i][j];
		}
		if (net->select_activate_function[i] == SOFTMAX) softmax(net, a, z, i);
	}

	for (j = 0; j < net->sizes[num_layers - 1]; j++)
	{
		output[j] = a[num_layers - 1][j];
		z_for_softmax[j] = z[num_layers - 2][j];
	}

	for (i = 0; i < num_layers; i++)
		free(a[i]);
	free(a);
	for (i = 0; i < num_layers - 1; i++)
		free(z[i]);
	free(z);
}

double QuadraticCost(Network* net, double** y, double** a, int n, double regularization_rate)
{
	int x, i, j, k;
	int num_layers = net->num_layers;
	double reg_term = 0, sum1 = 0, sum2 = 0;
	for (i = 0; i < num_layers; i++)
		for (j = 0; j < net->sizes[i + 1]; j++)
			for (k = 0; k < net->sizes[i]; k++)
				reg_term += net->weights[i][j][k] * net->weights[i][j][k];
	reg_term *= (regularization_rate / (2.0*n));

	for (x = 0; x < n; x++)
	{
		sum1 = 0;
		for (i = 0; i < net->sizes[num_layers - 1]; i++)
			sum1 += (y[x][i] - a[x][i])*(y[x][i] - a[x][i]);
		sum2 += sqrt(sum1)*sqrt(sum1);
	}
	sum2 *= 1.0 / (2 * n);
	sum2 += reg_term;
	return sum2;
}

double CrossEntropyCost(Network* net, double** y, double** z_outlayer, double** a, int n, double regularization_rate)
{
	int x, i, j, k;
	int num_layers = net->num_layers;
	double reg_term = 0, sum = 0;
	for (i = 0; i < num_layers; i++)
		for (j = 0; j < net->sizes[i + 1]; j++)
			for (k = 0; k < net->sizes[i]; k++)
				reg_term += net->weights[i][j][k] * net->weights[i][j][k];
	reg_term *= (regularization_rate / (2.0*n));

	for (x = 0; x < n; x++)
		for (i = 0; i < net->sizes[num_layers - 1]; i++)
		{
			//Softmax를 사용했을 때 log(0)으로 인한 underflow를 막기 위해 수정됨
			if (net->select_activate_function[num_layers - 2] == SOFTMAX)
			{
				sum -= y[x][i] * log_softmax(net, z_outlayer[x][i], z_outlayer[x]);
			}
			else sum -= (y[x][i] * log(a[x][i]));
		}

	sum = (1.0 / n) * sum + reg_term;
	return sum;
}

void cost_derivate(Network* net, double* output_activated, double* y, double* delta)
{
	int i;
	int output_neuron_size = net->sizes[net->num_layers - 1];
	for (i = 0; i < output_neuron_size; i++)
	{
		delta[i] = (output_activated[i] - y[i]);
	}
}

double sigmoid(double z)
{
	return 1.0 / (1.0 + exp(-z));
}
double sigmoid_prime(double z)
{
	return sigmoid(z)*(1 - sigmoid(z));
}
double ReLU(double z)
{
	if (z > 0) return z;
	else return 0;
}
double ReLU_prime(double z)
{
	if (z > 0) return 1;
	else return 0;
}
double tanh_activate(double z)
{
	return (1 + tanh(z / 2)) / 2.0;
}
double tanh_prime(double z)
{
	return (1 - tanh(z / 2)*tanh(z / 2)) / 2.0;
}
void softmax(Network *net, double** a, double** z, int i)
{
	//overflow, underflow를 막기 위해 수정됨
	int j;
	double sum = 0, max=0;
	for (j = 0; j < net->sizes[i + 1]; j++)
		if (max < z[i][j]) max = z[i][j];
	for (j = 0; j < net->sizes[i + 1]; j++)
		sum += exp(z[i][j] - max);
	for (j = 0; j < net->sizes[i + 1]; j++)
		a[i + 1][j] = exp(z[i][j] - max) / sum;
}
double log_softmax(Network *net, double z, double* z_outlayer)
{
	//overflow, underflow를 막기 위해 작성됨
	int j;
	int num_layers = net->num_layers;
	double sum = 0, max = 0;
	for (j = 0; j < net->sizes[num_layers - 1]; j++)
		if (max < z_outlayer[j]) max = z_outlayer[j];
	for (j = 0; j < net->sizes[num_layers - 1]; j++)
		sum += exp(z_outlayer[j] - max);
	return (z - max) - log(sum);
}
double softmax_prime(double a)
{
	return a * (1 - a);
}