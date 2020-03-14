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

	return 1;
}
