#include "neural_net.hpp"
#include "data_reader.hpp"
#include "options.hpp"
#include "eigen-3.3.7/Eigen/Dense"
#include "eigen-3.3.7/unsupported/Eigen/MatrixFunctions"

#include <iostream>
#include <vector>
#include <random>
#include <functional>

neural_net::neural_net(std::string path, std::vector<int> hidden_layer_sizes) : data(path) {
	
	//Need data initialized before this
	inputs = Eigen::MatrixXd(data.size(), data.rows() * data.cols());
	//Initialize the architecture
	arch.push_back(data.rows() * data.cols());
	for (int i = 0; i < hidden_layer_sizes.size(); i++)
		arch.push_back(hidden_layer_sizes[i]);
	arch.push_back(10);

	for (int i = 0; i < data.size(); i++)
		inputs.row(i) = Eigen::VectorXd::Map(&data.m_images[i][0], data.m_images[i].size());

	create_arch();
	feed_forward();
	back_propagation();
}

neural_net::~neural_net()
{
	//empty
}

void neural_net::create_arch()
{
	w1 = Eigen::MatrixXd::Random(arch[0], arch[1]);
	b1 = Eigen::MatrixXd::Random(1, arch[1]);
	w2 = Eigen::MatrixXd::Random(arch[1], arch[2]);
	b2 = Eigen::MatrixXd::Random(1, arch[2]);
}

void neural_net::feed_forward()
{
	Eigen::MatrixXd a1 = inputs * w1;
	for (int i = 0; i < a1.rows(); i++)
		a1.row(i) += b1.row(0);
	l1 = 1 / (1 + (a1.array() * -1).exp());

	Eigen::MatrixXd a2 = (l1 * w2);// + (b2 * l1);
	for (int i = 0; i < a2.rows(); i++)
		a2.row(i) += b2.row(0);
	Eigen::MatrixXd a2_exp = a2.array().exp();
	Eigen::MatrixXd a2_exp_sums = a2_exp.rowwise().sum();
	
	l2 = Eigen::MatrixXd(a2.rows(), a2.cols());
	for (int i = 0; i < a2.rows(); i++)
		l2.row(i) = a2_exp.row(i) / a2_exp_sums(i, 0);

}

void neural_net::back_propagation()
{
	model_error = get_error();
	Eigen::MatrixXd error_gradient = get_error_gradient();
	
	out_delta = (l1.transpose() * error_gradient);
	out_bias_delta = (error_gradient.colwise().sum() / error_gradient.rows());

	Eigen::MatrixXd hidden_output_error = error_gradient * w2.transpose();
	Eigen::MatrixXd sigmoid_prime = (l1.array() * (1 - l1.array())).matrix();
	Eigen::MatrixXd hidden_error = (hidden_output_error.array() * inputs.array() * sigmoid_prime.array()).matrix();
	hidden_delta = inputs.transpose() * hidden_error;
	hidden_bias_delta = (hidden_error.colwise().sum() / hidden_error.rows());
}

int neural_net::get_error()
{
	reshaped_target = Eigen::MatrixXd::Zero(l2.rows(), l2.cols());
	for (int i = 0; i < data.m_labels.size(); i++)
		reshaped_target(i, data.m_labels[i]) = 1;
	
	Eigen::MatrixXd temp = (l2.array() + 0.000000001).matrix();
	Eigen::MatrixXd model_error = reshaped_target *  (temp.array().log()).matrix();
	return -model_error.sum();
}

Eigen::MatrixXd neural_net::get_error_gradient()
{
	return (l2.array() - reshaped_target.array()).matrix();
}
