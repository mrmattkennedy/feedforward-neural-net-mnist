#ifndef neuralnet
#define neuralnet

#include "options.hpp"
#include "data_reader.hpp"
#include <vector>
#include <tuple>
#include <chrono>
#include <thrust/device_vector.h>
#include <cublas_v2.h>

class neural_net
{
	private:
		data_reader data;
		options opts;
		cublasHandle_t h;

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
		thrust::device_vector<float> labels;
		thrust::device_vector<float> shuffled_x;
		thrust::device_vector<float> shuffled_y;
		thrust::device_vector<float> test_in;
		thrust::device_vector<float> test_labels;
		thrust::device_vector<float> batch_x;
		thrust::device_vector<float> batch_y;
		thrust::device_vector<float> l1;
		thrust::device_vector<float> l2;
		thrust::device_vector<float> l3;
		
		thrust::device_vector<float> l3_delta;
		thrust::device_vector<float> l3_bias_delta;
		thrust::device_vector<float> l2_delta;
		thrust::device_vector<float> l2_bias_delta;
		thrust::device_vector<float> l1_delta;
		thrust::device_vector<float> l1_bias_delta;

		double model_error;
		const int train_size = 60000;
		const int test_size = 10000;
		const int blockSize = 256;
		const int NORMAL = 0x01;
		const int TRANSPOSE_A = 0x02;
		const int TRANSPOSE_B = 0x03;
		const float MAX_E = 88;
		
	public:
		neural_net(std::string base_path);
		~neural_net();
		
		void reset();
		void shuffle();
		std::tuple<std::tuple<int, __int64>, std::vector<std::tuple<int, float>>> neural_net::train(int batch_size=0);
		void create_arch();
		thrust::device_vector<float> neural_net::init_weight(int insize, int outsize);
		void feed_forward(thrust::device_vector<float> in);
		thrust::device_vector<float> clip(thrust::device_vector<float>, int max_power_val=88);
		void back_propagation();
		double get_error();
		thrust::device_vector<float> get_error_gradient();
		double get_accuracy();
		thrust::device_vector<float> matrix_multiply(thrust::device_vector<float> A, thrust::device_vector<float> B, int a_rows, int a_cols, int b_rows, int b_cols, int op);
};
#endif
