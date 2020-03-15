#ifndef NN
#define NN
#include "eigen-3.3.7/Eigen/Dense"
class neural_net:
	private:
		Eigen::MatrixXd w1;
		Eigen::MatrixXd b1;
		Eigen::MatrixXd w2;
		Eigen::MatrixXd b2;
		
		Eigen::MatrixXi inputs;
		Eigen::MatrixXd l1;
		Eigen::MatrixXd l2;
		
	public:
#endif
