#include <iostream>
#include "options.hpp"
#include "data_reader.hpp"
#include "eigen-3.3.7/Eigen/Dense"

int main(int argc, char** argv)
{
	options opts;
	std::string base_path = "..\\MNIST data\\";
	data_reader data(base_path);
	
	Eigen::MatrixXd inputs(data.size(), data.rows() * data.cols());
	for (int i = 0; i < data.size(); i++)
		inputs.row(i) = Eigen::VectorXd::Map(&data.m_images[i][0], data.m_images[i].size());
	std::cout << inputs.row(0) << std::endl;
	return 1;
}
