#ifndef neuralnet
#define neuralnet

#include "eigen-3.3.7/Eigen/Dense"
#include "options.hpp"
#include "data_reader.hpp"
#include <vector>

class neural_net
{
	private:
		std::vector<int> arch;
		data_reader data;
		options opts;

		Eigen::MatrixXd w1;
		Eigen::MatrixXd b1;
		Eigen::MatrixXd w2;
		Eigen::MatrixXd b2;
		
		Eigen::MatrixXd inputs;
		Eigen::MatrixXd l1;
		Eigen::MatrixXd l2;
		
		Eigen::MatrixXd reshaped_target;
		int model_error;

		Eigen::MatrixXd out_delta;
		Eigen::MatrixXd out_bias_delta;
		Eigen::MatrixXd hidden_delta;
		Eigen::MatrixXd hidden_bias_delta;
		
	public:
		neural_net(std::string base_path, std::vector<int> hidden_layer_sizes);
		~neural_net();

		void create_arch();
		void feed_forward();
		void back_propagation();
		int get_error();
		Eigen::MatrixXd get_error_gradient();
};
#endif
