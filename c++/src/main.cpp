#include <iostream>
#include "options.h"
#include "eigen-3.3.7/Eigen/Dense"
	
int main(int argc, char** argv)
{
	options opts;
	std::cout << opts.alpha << std::endl;
	Eigen::MatrixXd m = Eigen::MatrixXd::Random(3,3);
	std::cout << "m = " << std::endl << m << std::endl;
	return 1;
}
