#ifndef neuralnet
#define neuralnet

#include "eigen-3.3.7/Eigen/Dense"
#include "options.hpp"
#include "data_reader.hpp"
#include <vector>

class neural_net
{
	private:
		data_reader data;
		options opts;

		Eigen::MatrixXd w1;
		Eigen::MatrixXd b1;
		Eigen::MatrixXd w2;
		Eigen::MatrixXd b2;
		Eigen::MatrixXd w3;
		Eigen::MatrixXd b3;
		
		Eigen::MatrixXd v_w1;
		Eigen::MatrixXd v_b1;
		Eigen::MatrixXd v_w2;
		Eigen::MatrixXd v_b2;
		Eigen::MatrixXd v_w3;
		Eigen::MatrixXd v_b3;

		Eigen::MatrixXd inputs;
		Eigen::MatrixXd l1;
		Eigen::MatrixXd l2;
		Eigen::MatrixXd l3;
		
		Eigen::MatrixXd reshaped_target;
		Eigen::MatrixXd test_target;
		int model_error;

		Eigen::MatrixXd out_delta;
		Eigen::MatrixXd out_bias_delta;
		Eigen::MatrixXd hidden_2_delta;
		Eigen::MatrixXd hidden_2_bias_delta;
		Eigen::MatrixXd hidden_delta;
		Eigen::MatrixXd hidden_bias_delta;
		
	public:
		neural_net(std::string base_path);
		~neural_net();
		
		void reset();
		std::tuple<std::tuple<int, __int64>, std::vector<std::tuple<int, float>>> train(int batch_size=0);
		void create_arch();
		void feed_forward(Eigen::MatrixXd in);
		void back_propagation();
		int get_error();
		Eigen::MatrixXd get_error_gradient();
		double get_accuracy();
};
#endif
