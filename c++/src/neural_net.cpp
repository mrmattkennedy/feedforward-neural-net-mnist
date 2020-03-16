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
}

neural_net::~neural_net()
{
	//empty
}

void neural_net::create_arch()
{
	/*
	std::vector<std::vector<double>> w1_init = init_weights(arch[0], arch[1]);
	w1 = Eigen::MatrixXd(arch[0], arch[1]);
	for (int i = 0; i < arch[0]; i++)
		w1.row(i) = Eigen::VectorXd::Map(&w1_init[i][0], w1_init[i].size());
	
	//Doing 1 row is incredibly slow, need n rows by 1 column
	
	std::vector<std::vector<double>> b1_init = init_weights(arch[1], 1);
	b1 = Eigen::MatrixXd(arch[1], 1);
	for (int i = 0; i < arch[1]; i++)
		b1.row(i) = Eigen::VectorXd::Map(&b1_init[i][0], b1_init[i].size());
	
	b1 = Eigen::VectorXd(arch[1]);
	std::vector<std::vector<double>> w2_init = init_weights(arch[1], arch[2]);
	w2 = Eigen::MatrixXd(arch[1], arch[2]);
	for (int i = 0; i < arch[1]; i++)
		w2.row(i) = Eigen::VectorXd::Map(&w2_init[i][0], w2_init[i].size());

	std::vector<std::vector<double>> b2_init = init_weights(arch[2], 1);
	b2 = Eigen::MatrixXd(arch[2], 1);
	for (int i = 0; i < arch[2]; i++)
		b2.row(i) = Eigen::VectorXd::Map(&b2_init[i][0], b2_init[i].size());

	w1_init.clear();
	b1_init.clear();
	w2_init.clear();
	b2_init.clear();
	*/

	w1 = Eigen::MatrixXd::Random(arch[0], arch[1]);
	b1 = Eigen::MatrixXd::Random(1, arch[1]);
	w2 = Eigen::MatrixXd::Random(arch[1], arch[2]);
	b2 = Eigen::MatrixXd::Random(1, arch[2]);
}


std::vector<std::vector<double>> neural_net::init_weights(int from_size, int to_size)
{
	double lower_bound = 0;
	double upper_bound = 1;
	std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
	std::default_random_engine re;

	std::vector<std::vector<double>> w_init;
	for (int i = 0; i < from_size; i++)
	{	
		std::vector<double> temp(to_size);
		for (int j = 0; j < to_size; j++)
		{
			double weight = unif(re);
			temp[j] = weight;
		}
		w_init.push_back(temp);
	}
	return w_init;
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
//	std::cout << l2.row(0) << std::endl << "Sum is " << l2.row(0).sum() << std::endl;

}

double neural_net::sigmoid(double z)
{
	//z = std::max(-500, std::min(z, 500));
	//return 1 / (1 + z);
	return z;
}
