#ifndef neuralnet
#define neuralnet

#include "options.hpp"
#include "data_reader.hpp"
#include <vector>

class neural_net
{
	private:
		data_reader data;
		options opts;

		int model_error;

		
	public:
		neural_net(std::string base_path);
		~neural_net();
		
		void train();
		void create_arch();
		void feed_forward();
		void back_propagation();
		int get_error();
		double get_accuracy();
};
#endif
