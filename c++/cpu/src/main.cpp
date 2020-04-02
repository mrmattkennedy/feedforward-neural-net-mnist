#include "neural_net.hpp"
#include <vector>
#include <iostream>
#include "eigen-3.3.7/Eigen/Dense"
#include "eigen-3.3.7/unsupported/Eigen/MatrixFunctions"

int main(int argc, char** argv)
{
	std::string base_path = "..\\..\\MNIST data\\";
	neural_net nn(base_path);
	nn.train();
	Eigen::MatrixXf mat(2, 2);
	mat << 1, 2, 3, 4;
	Eigen::MatrixXf mat2(2, 2);
	mat2 << 1, 2, 3, 4;
	//std::cout << "here" << std::endl;
	Eigen::MatrixXf mat3 = (mat.array() * mat2.array()).matrix();
	//std::cout << mat3 << std::endl;

	return 1;
}
