#include "neural_net.hpp"
#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>

int main(int argc, char** argv)
{
	std::string base_path = "..\\..\\MNIST data\\";
	neural_net nn(base_path);
	//Read in batche sizes from file and remove spaces
	std::ifstream batch_sizes("data\\batch_sizes.data");
	std::string line;
	std::getline(batch_sizes, line);
	line.erase(std::remove(line.begin(), line.end(), ' '), line.end()); 

	//Deliminate and pushback on vector
	std::istringstream ss(line);
	std::string str_size;
	std::vector<int> sizes;

	while (std::getline(ss, str_size, ','))
		sizes.push_back(std::stoi(str_size));
	
	//Run each batch size	
	std::vector<std::vector<std::tuple<int, float>>> accuracies;
	std::vector<std::tuple<int, __int64>> times;
	for (auto &batch_size : sizes)
		if (batch_size >= 30)
		{
			auto result = nn.train(batch_size);
			times.push_back(std::get<0>(result));
			accuracies.push_back(std::get<1>(result));
		}
	
	//Save accuracy results
	std::ofstream acc_file("data\\gpu_results.dat");
	for (auto &batch : accuracies)
	{
		for (auto &row : batch)
			acc_file << std::get<0>(row) << "," << std::get<1>(row) << std::endl;
		acc_file << std::endl;
	}
	acc_file.close();

	//Save time results
	std::ofstream time_file("data\\gpu_times.dat");
	for (auto &time : times)
		time_file << std::get<0>(time) << "," << std::get<1>(time)/1000.0f << std::endl;
	time_file.close();

	return 0;
}
