#include <iostream>
#include "options.hpp"
#include "data_reader.hpp"
#include "eigen-3.3.7/Eigen/Dense"

int main(int argc, char** argv)
{
	options opts;
	std::string base_path = "..\\MNIST data\\";
	data_reader data(base_path);
	
	Eigen::MatrixXi inputs(data.size(), data.rows() * data.cols());
	for (int i = 0; i < data.size(); i++)
		inputs.row(i) = Eigen::VectorXi::Map(&data.m_images[i][0], data.m_images[i].size());


	return 1;
}
