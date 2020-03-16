#include "neural_net.hpp"
#include <vector>

int main(int argc, char** argv)
{
	std::string base_path = "..\\MNIST data\\";
	std::vector<int> temp;
	temp.push_back(200);
	neural_net nn(base_path, temp);
	return 1;
}
