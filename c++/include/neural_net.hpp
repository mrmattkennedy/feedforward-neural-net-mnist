#ifndef neuralnet
#define neuralnet

#include "eigen-3.3.7/Eigen/Dense"
#include "options.hpp"
#include "data_reader.hpp"
#include <vector>

class neural_net
{
	private:
		Eigen::MatrixXd w1;
		Eigen::MatrixXd b1;
		Eigen::MatrixXd w2;
		Eigen::MatrixXd b2;
		
		Eigen::MatrixXd inputs;
		Eigen::MatrixXd l1;
		Eigen::MatrixXd l2;
		
		std::vector<int> arch;
		data_reader data;
		options opts;
		
	public:
		neural_net(std::string base_path, std::vector<int> hidden_layer_sizes);
		~neural_net();

		void create_arch();
		std::vector<std::vector<double>> init_weights(int from_size, int to_size);
		void feed_forward();
		double sigmoid(double z);
};
#endif
