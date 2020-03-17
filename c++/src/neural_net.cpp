#include "neural_net.hpp"
#include "data_reader.hpp"
#include "options.hpp"
#include "eigen-3.3.7/Eigen/Dense"
#include "eigen-3.3.7/unsupported/Eigen/MatrixFunctions"

#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <algorithm>
#include <chrono>

neural_net::neural_net(std::string path, std::vector<int> hidden_layer_sizes) : data(path) {
	
	//Need data initialized before this
	inputs = Eigen::MatrixXd(data.size(), data.rows() * data.cols());
	//Initialize the architecture
	arch.push_back(data.rows() * data.cols());
	for (int i = 0; i < hidden_layer_sizes.size(); i++)
		arch.push_back(hidden_layer_sizes[i]);
	arch.push_back(10);
	
	/*
	for (int i = 0; i < data.size(); i++)
		inputs.row(i) = Eigen::VectorXd::Map(&data.m_images[i][0], data.m_images[i].size());
	
	reshaped_target = Eigen::MatrixXd::Zero(data.size(), 10);
	for (int i = 0; i < data.size(); i++)
		reshaped_target(i, data.m_labels[i]) = 1;
	*/
}

neural_net::~neural_net()
{
	//empty
}

void neural_net::train()
{
	create_arch();
	int train_size = 60000;
	//Work on creating batches with random array of arange size 70000. Passing to matrix as (array, all) gives random
	std::vector<int> shuffle_vector(train_size);
	std::iota(shuffle_vector.begin(), shuffle_vector.end(), 0);

	for (int i = 0; i < opts.epochs; i++)
	{	
		//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    		//std::default_random_engine e(seed);
		std::random_shuffle(shuffle_vector.begin(), shuffle_vector.end());

		opts.alpha *= (1 / (1 + opts.decay * i));
		for (int j = 0; j < opts.batches; j++)
		{
			inputs.resize(0, 0);
			inputs = Eigen::MatrixXd(opts.batch_size, data.rows() * data.cols());
			reshaped_target.resize(0, 0);
			reshaped_target = Eigen::MatrixXd::Zero(opts.batch_size, opts.n_o);
			for (int i = j * opts.batch_size; i < (j * opts.batch_size) + opts.batch_size; i++)
			{
				inputs.row(i - (j * opts.batch_size)) = Eigen::VectorXd::Map(&data.m_images[shuffle_vector[i]][0], data.m_images[shuffle_vector[i]].size());
				reshaped_target(i -  (j * opts.batch_size), data.m_labels[shuffle_vector[i]]) = 1;
			}
			feed_forward();
			back_propagation();
			
			v_w2.noalias() = (opts.beta * v_w2) + ((1 - opts.beta) * out_delta);
			v_b2.noalias() = (opts.beta * v_b2) + ((1 - opts.beta) * out_bias_delta);
			v_w1.noalias() = (opts.beta * v_w1) + ((1 - opts.beta) * hidden_delta);
			v_b1.noalias() = (opts.beta * v_b1) + ((1 - opts.beta) * hidden_bias_delta);

			w2.noalias() -= (opts.alpha * v_w2);
			b2.noalias() -= (opts.alpha * v_b2);
			w1.noalias() -= (opts.alpha * v_w1);
			b1.noalias() -= (opts.alpha * v_b1);
			/*
			w2.noalias() -= (opts.alpha * out_delta);
			b2.noalias() -= (opts.alpha * out_bias_delta);
			w1.noalias() -= (opts.alpha * hidden_delta);
			b1.noalias() -= (opts.alpha * hidden_bias_delta);
			*/
		}
		if (i % 1 == 0)
			printf("Epoch %5d\tloss: %5d\taccuracy: %.4f\n", i, model_error, get_accuracy());	
	}
		
}

void neural_net::create_arch()
{
	w1 = Eigen::MatrixXd::Random(arch[0], arch[1]);
	b1 = Eigen::MatrixXd::Random(1, arch[1]);
	w2 = Eigen::MatrixXd::Random(arch[1], arch[2]);
	b2 = Eigen::MatrixXd::Random(1, arch[2]);

	v_w1 = Eigen::MatrixXd::Zero(arch[0], arch[1]);
	v_b1 = Eigen::MatrixXd::Zero(1, arch[1]);
	v_w2 = Eigen::MatrixXd::Zero(arch[1], arch[2]);
	v_b2 = Eigen::MatrixXd::Zero(1, arch[2]);
}

void neural_net::feed_forward()
{
	Eigen::MatrixXd a1 = inputs * w1;
	for (int i = 0; i < a1.rows(); i++)
		a1.row(i) += b1.row(0);
	l1.noalias() = a1.unaryExpr(
			[](const double& x) {
				return 1.0 / (1.0 + std::exp(-std::max(-500.0, std::min(x, 500.0))));
	});
	Eigen::MatrixXd a2;
	a2.noalias() = (l1 * w2);
	for (int i = 0; i < a2.rows(); i++)
		a2.row(i) += b2.row(0);
	Eigen::MatrixXd a2_exp = a2.array().exp();
	Eigen::MatrixXd a2_exp_sums = a2_exp.rowwise().sum();
	
	l2.noalias() = Eigen::MatrixXd(a2.rows(), a2.cols());
	for (int i = 0; i < a2.rows(); i++)
		l2.row(i) = a2_exp.row(i) / a2_exp_sums(i, 0);

	a1.resize(0, 0);
	a2.resize(0, 0);
	a2_exp.resize(0, 0);
	a2_exp_sums.resize(0, 0);
}

void neural_net::back_propagation()
{
	model_error = get_error();
	Eigen::MatrixXd error_gradient = get_error_gradient();
	
	out_delta = (l1.transpose() * error_gradient) / error_gradient.rows();
	out_bias_delta = (error_gradient.colwise().sum() / error_gradient.rows());
	Eigen::MatrixXd hidden_output_error = error_gradient * w2.transpose();
	Eigen::MatrixXd sigmoid_prime = l1.unaryExpr(
			[](const double& x) {
				return x * (1 - x);
	});
	Eigen::MatrixXd hidden_error = (hidden_output_error.array() * l1.array() * sigmoid_prime.array()).matrix();

	hidden_delta = inputs.transpose() * hidden_error;
	hidden_bias_delta = (hidden_error.colwise().sum() / hidden_error.rows());

	error_gradient.resize(0, 0);
	hidden_output_error.resize(0, 0);
	sigmoid_prime.resize(0, 0);
	hidden_error.resize(0, 0);
}

int neural_net::get_error()
{	
	Eigen::MatrixXd temp = (l2.array() + 0.000000001).matrix();
	Eigen::MatrixXd err;
	err.noalias() = reshaped_target * (temp.array().log()).matrix();
	return -err.sum();
}

Eigen::MatrixXd neural_net::get_error_gradient()
{
	Eigen::MatrixXd temp = l2;
	temp.noalias() -= reshaped_target;
	return temp;
}

double neural_net::get_accuracy()
{
	double accuracy = 0.0;
	Eigen::MatrixXf::Index maxIndex;
	Eigen::MatrixXf::Index target;
	for (int i = 0; i < l2.rows(); i++)
	{
		l2.row(i).maxCoeff(&maxIndex);
		reshaped_target.row(i).maxCoeff(&target);
		if (maxIndex == target)
			accuracy +=1;
	}
	return accuracy / l2.rows();
}
