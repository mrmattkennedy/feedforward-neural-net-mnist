#ifndef neuralnet
#define neuralnet

#include "options.hpp"
#include "data_reader.hpp"
#include <vector>
#include <thrust/device_vector.h>

class neural_net
{
	private:
		data_reader data;
		options opts;

		thrust::device_vector<float> w1;
		thrust::device_vector<float> b1;
		thrust::device_vector<float> w2;
		thrust::device_vector<float> b2;
		thrust::device_vector<float> w3;
		thrust::device_vector<float> b3;

		thrust::device_vector<float> v_w1;
		thrust::device_vector<float> v_b1;
		thrust::device_vector<float> v_w2;
		thrust::device_vector<float> v_b2;
		thrust::device_vector<float> v_w3;
		thrust::device_vector<float> v_b3;

		thrust::device_vector<float> inputs;
		thrust::device_vector<float> l1;
		thrust::device_vector<float> l2;
		thrust::device_vector<float> l3;
		thrust::device_vector<float> labels;

		double model_error;

		
	public:
		neural_net(std::string base_path);
		~neural_net();
		
		void train();
		void create_arch();
		thrust::device_vector<float> neural_net::init_weight(int insize, int outsize);
		void feed_forward();
		thrust::device_vector<float> clip(thrust::device_vector<float>, int max_power_val=88);
		void back_propagation();
		double get_error();
		thrust::device_vector<float> get_error_gradient();
		double get_accuracy();
};
#endif
