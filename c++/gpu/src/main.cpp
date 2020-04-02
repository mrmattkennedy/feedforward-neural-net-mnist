#include "neural_net.hpp"


int main(int argc, char** argv)
{
	std::string base_path = "..\\..\\MNIST data\\";
	neural_net nn(base_path);
	nn.train();
	
	return 0;
}
