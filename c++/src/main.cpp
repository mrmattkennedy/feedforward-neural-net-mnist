#include <iostream>
#include "options.hpp"
#include "data_reader.hpp"
#include "eigen-3.3.7/Eigen/Dense"

int main(int argc, char** argv)
{
	options opts;
	std::string base_path = "..\\MNIST data\\";
	//Capacity > size
	data_reader data(base_path);
	
	//constexpr int num_rows = data.size();
	//constexpr int num_cols = data.rows() * data.cols();
	//printf("%d, %d", num_rows, num_cols);
	Eigen::MatrixXi test = Eigen::Map<Eigen::Matrix<int, 1, 784 > >(data.m_images[0].data());
	return 1;
}
