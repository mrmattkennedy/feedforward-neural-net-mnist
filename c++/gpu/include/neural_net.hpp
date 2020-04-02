#ifndef neuralnet
#define neuralnet

#include "options.hpp"
#include "data_reader.hpp"
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/generate.h>
#include <thrust/random.h>

class neural_net
{
	private:
		data_reader data;
		options opts;

		std::vector<thrust::device_vector<float>> w1;
		std::vector<thrust::device_vector<float>> b1;
		std::vector<thrust::device_vector<float>> w2;
		std::vector<thrust::device_vector<float>> b2;
		std::vector<thrust::device_vector<float>> w3;
		std::vector<thrust::device_vector<float>> b3;

		std::vector<thrust::device_vector<float>> v_w1;
		std::vector<thrust::device_vector<float>> v_b1;
		std::vector<thrust::device_vector<float>> v_w2;
		std::vector<thrust::device_vector<float>> v_b2;
		std::vector<thrust::device_vector<float>> v_w3;
		std::vector<thrust::device_vector<float>> v_b3;

		int model_error;

		
	public:
		neural_net(std::string base_path);
		~neural_net();
		
		void train();
		void create_arch();
		std::vector<thrust::device_vector<float>> init_weight(int insize, int outsize);
		std::vector<thrust::device_vector<float>> init_velocity(int insize, int outsize);
		void feed_forward();
		void back_propagation();
		int get_error();
		double get_accuracy();
};
#endif
