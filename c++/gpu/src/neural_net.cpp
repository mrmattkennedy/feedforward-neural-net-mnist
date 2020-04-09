#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <thrust/equal.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <cstdlib>
#include <math.h>

#include "neural_net.hpp"
#include "data_reader.hpp"
#include "options.hpp"

struct RandGen
{
	//N is size of vector, M is iteration
	unsigned int N;
	RandGen(unsigned int _N) : N(_N) {};

	__device__ float operator () (unsigned int thread_id)
	{
		thrust::default_random_engine randEng;
		randEng.discard(N * thread_id);
		thrust::uniform_real_distribution<float> uniDist(-0.2, 0.2);
		return uniDist(randEng);
	}
};


__global__ 
void sigmoid_cuda(int n, float *x)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		x[i] = 1 / (1 + __expf(-x[i]));
}

__global__ 
void sigmoid_prime_cuda(int n, float *outputs, float *gradient)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		gradient[i] = outputs[i] * (1 - outputs[i]);
}

__global__ 
void cuda_get_exp(int n, float *x)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		x[i] = __expf(x[i]);
}


__global__ 
void softmax_cuda(int n, float *e, float *e_sums, int n_o)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		e[i] = e[i] / e_sums[i / n_o];
}

__global__ 
void cuda_get_error(int n, float *outputs, float *labels, double *sums, int n_o)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
	{
		sums[i] = -__logf(outputs[(i*n_o) + (int)labels[i]] + 1e-10);
	}
}

__global__ 
void cuda_get_error_gradient(int n, float *outputs, float *labels, float *gradient, int n_o)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		if (i == ((i / n_o) + labels[i / n_o]))
			gradient[i] = outputs[i] - 1;
		else
			gradient[i] = outputs[i];
}


__global__ 
void cuda_get_bias_delta(int n, float *gradient, float *layer_bias, int num_nodes)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		layer_bias[i % num_nodes] += gradient[i] / n;
}


__global__ 
void cuda_get_layer_error(int n, float *out_error, float *outputs, float *out_gradient, float *layer_error)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		layer_error[i] = out_error[i] * outputs[i] * out_gradient[i];
}

__global__ 
void cuda_update_velocity(int n, float *v, float *delta, float beta)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		v[i] = (beta * v[i]) + ((1 - beta) * delta[i]);
}

__global__ 
void cuda_update_weight(int n, float *w, float *v, float alpha)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		w[i] -= alpha * v[i];
}


neural_net::neural_net(std::string path) : data(path) 
{
	//empty	
}

neural_net::~neural_net()
{
	//empty
}

void neural_net::train()
{
	clock_t start, end;
	create_arch();
	int train_size = 60000;
	
	inputs = thrust::device_vector<float>(data.m_images.size());
	thrust::copy(data.m_images.begin(), data.m_images.end(), inputs.begin());
	labels = thrust::device_vector<float>(data.m_labels.size());
	thrust::copy(data.m_labels.begin(), data.m_labels.end(), labels.begin());
	
	cublasCreate(&h);
	start = clock();
	
	int blockSize = 256;
	int numBlocks = (opts.n_o * opts.n_h2 + blockSize - 1) / blockSize;

	for (int i = 0; i < 30; i++)
	{
		feed_forward();
		back_propagation();
		printf("V3 before:\t");
		for (int i = 0; i < 10; i++)
			std::cout << v_w3[i] << " ";
		std::cout << std::endl;
		//Update velocities
		cuda_update_velocity<<<numBlocks, blockSize>>>(opts.n_o * opts.n_h2, 
				thrust::raw_pointer_cast(v_w3.data()),
				thrust::raw_pointer_cast(l3_delta.data()),
				opts.beta);
		cudaDeviceSynchronize(); 
		printf("V3 after:\t");
		for (int i = 0; i < 10; i++)
			std::cout << v_w3[i] << " ";
		std::cout << std::endl;

		numBlocks = (opts.n_h2 * opts.n_h1 + blockSize - 1) / blockSize;
		printf("V2 before:\t");
		for (int i = 0; i < 10; i++)
			std::cout << v_w2[i] << " ";
		std::cout << std::endl;
		cuda_update_velocity<<<numBlocks, blockSize>>>(opts.n_h2 * opts.n_h1, 
				thrust::raw_pointer_cast(v_w2.data()),
				thrust::raw_pointer_cast(l2_delta.data()),
				opts.beta);
		cudaDeviceSynchronize(); 
		printf("V2 after:\t");
		for (int i = 0; i < 10; i++)
			std::cout << v_w2[i] << " ";
		std::cout << std::endl;

		numBlocks = (opts.n_h1 * opts.n_x + blockSize - 1) / blockSize;
		printf("V1 before:\t");
		for (int i = 0; i < 10; i++)
			std::cout << v_w1[i] << " ";
		std::cout << std::endl;
		cuda_update_velocity<<<numBlocks, blockSize>>>(opts.n_h1 * opts.n_x,
				thrust::raw_pointer_cast(v_w1.data()),
				thrust::raw_pointer_cast(l1_delta.data()),
				opts.beta);
		cudaDeviceSynchronize(); 
		printf("V1 after:\t");
		for (int i = 0; i < 10; i++)
			std::cout << v_w1[i] << " ";
		std::cout << std::endl;

		//Update weights
		numBlocks = (opts.n_o * opts.n_h2 + blockSize - 1) / blockSize;
		printf("W3 before:\t");
		for (int i = 0; i < 10; i++)
			std::cout << w3[i] << " ";
		std::cout << std::endl;
		cuda_update_weight<<<numBlocks, blockSize>>>(opts.n_o * opts.n_h2, 
				thrust::raw_pointer_cast(w3.data()),
				thrust::raw_pointer_cast(v_w3.data()),
				opts.alpha);
		cudaDeviceSynchronize(); 
		printf("W3 after:\t");
		for (int i = 0; i < 10; i++)
			std::cout << w3[i] << " ";
		std::cout << std::endl;

		numBlocks = (opts.n_h2 * opts.n_h1 + blockSize - 1) / blockSize;
		printf("W2 before:\t");
		for (int i = 0; i < 10; i++)
			std::cout << w2[i] << " ";
		std::cout << std::endl;
		cuda_update_weight<<<numBlocks, blockSize>>>(opts.n_h2 * opts.n_h1, 
				thrust::raw_pointer_cast(w2.data()),
				thrust::raw_pointer_cast(v_w2.data()),
				opts.alpha);
		cudaDeviceSynchronize(); 
		printf("W2 after:\t");
		for (int i = 0; i < 10; i++)
			std::cout << w2[i] << " ";
		std::cout << std::endl;

		numBlocks = (opts.n_h1 * opts.n_x + blockSize - 1) / blockSize;
		printf("W1 before:\t");
		for (int i = 0; i < 10; i++)
			std::cout << w1[i] << " ";
		std::cout << std::endl;
		cuda_update_weight<<<numBlocks, blockSize>>>(opts.n_h1 * opts.n_x, 
				thrust::raw_pointer_cast(w1.data()),
				thrust::raw_pointer_cast(v_w1.data()),
				opts.alpha);
		cudaDeviceSynchronize(); 
		printf("W1 after:\t");
		for (int i = 0; i < 10; i++)
			std::cout << w1[i] << " ";
		std::cout << std::endl;
		printf("%d:\tError:%f\tAccuracy:\n", i, model_error);
		std::cout << std::endl;
	}
	end = clock();
	cublasDestroy(h);
	double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
	printf("%f\n", time_taken);
		
}

void neural_net::create_arch()
{
	w1 = init_weight(opts.n_x, opts.n_h1);
	b1 = init_weight(1, opts.n_h1);
	w2 = init_weight(opts.n_h1, opts.n_h2);
	b2 = init_weight(1, opts.n_h2);
	w3 = init_weight(opts.n_h2, opts.n_o);
	b3 = init_weight(1, opts.n_o);

	v_w1 = thrust::device_vector<float>(opts.n_x * opts.n_h1, 0);
	v_b1 = thrust::device_vector<float>(1 * opts.n_h1, 0);
	v_w2 = thrust::device_vector<float>(opts.n_h1 * opts.n_h2, 0);
	v_b2 = thrust::device_vector<float>(1 * opts.n_h1, 0);
	v_w3 = thrust::device_vector<float>(opts.n_h2 * opts.n_o, 0);
	v_b3 = thrust::device_vector<float>(1 * opts.n_h1, 0);
}


thrust::device_vector<float> neural_net::init_weight(int insize, int outsize)
{

	thrust::device_vector<float> d_vec(insize * outsize);
	thrust::transform(
		thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(insize * outsize),
		d_vec.begin(),
		RandGen(insize * outsize));

	return d_vec;
}



void neural_net::feed_forward()
{

	//Using thrust transform with struct doesn't work for large vectors. Need to use cublas GEMM (general matrix multiply) algorithms.
//	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n*r), result.begin(), dp(thrust::raw_pointer_cast(inputs.data()), thrust::raw_pointer_cast(w1.data()), m, n, r));
	int blockSize = 256;
	
	//Dot product of inputs and w1
	int n = inputs.size() / opts.n_x, m = opts.n_x, r = opts.n_h1; //inputs is 70000x784 (NxM), w1 is 784x600 (MxR),
	auto a1 = matrix_multiply(inputs, w1, n, m, m, r, NORMAL);
	cudaDeviceSynchronize(); 
	//clip a1 values, assign to l1
	l1 = clip(a1);

	//Cuda kernel for __expf, fast exponent
	int numBlocks = (n*r + blockSize - 1) / blockSize;
	sigmoid_cuda<<<numBlocks, blockSize>>>(n*r, thrust::raw_pointer_cast(l1.data()));
	cudaDeviceSynchronize();
	
	
	//Hidden layer 2
	m = opts.n_h1, r = opts.n_h2;
	auto a2 = matrix_multiply(l1, w2, n, m, m, r, NORMAL);
	cudaDeviceSynchronize(); 
	
	//clip a2 values, assign to l2
	l2 = clip(a2);

	//Cuda kernel for __expf, fast exponent
	numBlocks = (n*r + blockSize - 1) / blockSize;
	sigmoid_cuda<<<numBlocks, blockSize>>>(n*r, thrust::raw_pointer_cast(l2.data()));
	cudaDeviceSynchronize();


	//Output layer
	m = opts.n_h2, r = opts.n_o;
	auto a3 = matrix_multiply(l2, w3, n, m, m, r, NORMAL);
	cudaDeviceSynchronize(); 
	
	//Get exponent
	l3 = a3;
	numBlocks = (n*r + blockSize - 1) / blockSize;
	cuda_get_exp<<<numBlocks, blockSize>>>(n*r, thrust::raw_pointer_cast(l3.data()));

	//Get sequences
	int sequence_length = opts.n_o;
	int sequences = l3.size() / sequence_length;
	thrust::device_vector<float> exp_sums(sequences);
	thrust::reduce_by_key(thrust::device, 
			thrust::make_transform_iterator(thrust::counting_iterator<int>(0), thrust::placeholders::_1 / sequence_length), 
			thrust::make_transform_iterator(thrust::counting_iterator<int>(sequences * sequence_length), thrust::placeholders::_1 / sequence_length), 
			l3.begin(), 
			thrust::discard_iterator<int>(), 
			exp_sums.begin());

	//Do softmax
	softmax_cuda<<<numBlocks, blockSize>>>(n*r, thrust::raw_pointer_cast(l3.data()), thrust::raw_pointer_cast(exp_sums.data()), opts.n_o);
	cudaDeviceSynchronize(); 
}


thrust::device_vector<float> neural_net::clip(thrust::device_vector<float> in, int max_power_val)
{
	thrust::device_vector<float>::iterator iter = thrust::max_element(in.begin(), in.end());
	float max_val = std::abs(*iter);
	iter = thrust::min_element(in.begin(), in.end());
	float min_val = std::abs(*iter);
	
	//Get largest after absolute value
	float largest_val = std::max(min_val, max_val);

	//Divide max_val by this factor, as e^88 overflows for 32 bit floats
	float factor = max_power_val / largest_val;
	
	//Multiply each element by factor
	thrust::device_vector<float> ret(in.size(), 0);
	thrust::transform(in.begin(), in.end(),
		thrust::make_constant_iterator(factor),
		ret.begin(),
		thrust::multiplies<float>());

	return ret;
}

void neural_net::back_propagation()
{	
	model_error = get_error();
	thrust::device_vector<float> error_gradient = get_error_gradient();
	
	int n = inputs.size() / opts.n_x, m = opts.n_h2, r = opts.n_o;
	int blockSize = 256;	
	int numBlocks = (n + blockSize - 1) / blockSize;
	
	//Output layer
	//Get out delta
	l3_delta = matrix_multiply(l2, error_gradient, n, m, n, r, TRANSPOSE_A);
	thrust::for_each(l3_delta.begin(), l3_delta.end(), thrust::placeholders::_1 /= n);

	//Get bias delta
	l3_bias_delta = thrust::device_vector<float>(r, 0);
	cuda_get_bias_delta<<<numBlocks, blockSize>>>(n,
			thrust::raw_pointer_cast(error_gradient.data()), 
			thrust::raw_pointer_cast(l3_bias_delta.data()), 
			r);
	
	//Hidden layer 2	
	//Get hidden layer 2 output error
	m = opts.n_o, r = opts.n_h2;
	auto l2_out_error = matrix_multiply(error_gradient, w3, n, m, r, m, TRANSPOSE_B);

	//Get sigmoid derivative for hidden layer 2
	thrust::device_vector<float> l2_sigmoid_prime(n*r, 0);
	sigmoid_prime_cuda<<<numBlocks, blockSize>>>(n,
			thrust::raw_pointer_cast(l2.data()), 
			thrust::raw_pointer_cast(l2_sigmoid_prime.data()));
	
	//Get hidden layer 2 error
	thrust::device_vector<float> l2_error(n*r, 0);
	cuda_get_layer_error<<<numBlocks, blockSize>>>(n,
			thrust::raw_pointer_cast(l2_out_error.data()), 
			thrust::raw_pointer_cast(l2.data()), 
			thrust::raw_pointer_cast(l2_sigmoid_prime.data()),
			thrust::raw_pointer_cast(l2_error.data()));
	
	//Get hidden layer 2 deltas
	m = opts.n_h1, r = opts.n_h2;
	l2_delta = matrix_multiply(l1, l2_error, n, m, n, r, TRANSPOSE_A);
	
	//Get bias delta
	l2_bias_delta = thrust::device_vector<float>(r, 0);
	cuda_get_bias_delta<<<numBlocks, blockSize>>>(n,
			thrust::raw_pointer_cast(l2_error.data()), 
			thrust::raw_pointer_cast(l2_bias_delta.data()), 
			r);
	

	//Hidden layer 1
	//Get hidden layer 1 output error
	auto l1_out_error = matrix_multiply(l2_error, w2, n, r, m, r, TRANSPOSE_B);
	

	//Get sigmoid derivative for hidden layer 1
	thrust::device_vector<float> l1_sigmoid_prime(n*m, 0);
	sigmoid_prime_cuda<<<numBlocks, blockSize>>>(n,
			thrust::raw_pointer_cast(l1.data()), 
			thrust::raw_pointer_cast(l1_sigmoid_prime.data()));
	
	//Get hidden layer 1 error
	thrust::device_vector<float> l1_error(n*m, 0);
	cuda_get_layer_error<<<numBlocks, blockSize>>>(n,
			thrust::raw_pointer_cast(l1_out_error.data()), 
			thrust::raw_pointer_cast(l1.data()), 
			thrust::raw_pointer_cast(l1_sigmoid_prime.data()),
			thrust::raw_pointer_cast(l1_error.data()));
	

	//Get hidden layer 1 deltas
	m = opts.n_x, r = opts.n_h1;
	l1_delta = matrix_multiply(inputs, l1_error, n, m, n, r, TRANSPOSE_A);
	
	//Get bias delta
	l1_bias_delta = thrust::device_vector<float>(r, 0);
	cuda_get_bias_delta<<<numBlocks, blockSize>>>(n,
			thrust::raw_pointer_cast(l1_error.data()), 
			thrust::raw_pointer_cast(l1_bias_delta.data()), 
			r);
	
	//printf("Size is %d, %d\n", l1_delta.size(), l1_bias_delta.size());
	printf("Deltas are: \n");
	printf("L3:\t");
	for (int i = 0; i < 10; i++)
		std::cout << l3_delta[i] << " ";
	std::cout << std::endl;
	printf("L3B:\t");
	for (int i = 0; i < 10; i++)
		std::cout << l3_bias_delta[i] << " ";
	std::cout << std::endl;
	printf("L2:\t");
	for (int i = 0; i < 10; i++)
		std::cout << l2_delta[i] << " ";
	std::cout << std::endl;
	printf("L2B:\t");
	for (int i = 0; i < 10; i++)
		std::cout << l2_bias_delta[i] << " ";
	std::cout << std::endl;
	printf("L1:\t");
	for (int i = 0; i < 10; i++)
		std::cout << l1_delta[i] << " ";
	std::cout << std::endl;
	printf("L1B:\t");
	for (int i = 0; i < 10; i++)
		std::cout << l1_bias_delta[i] << " ";
	std::cout << std::endl;
	/*
	printf("Sizes are %d, %d, %d, %d\n", l1_out_error.size(), l1_sigmoid_prime.size(), l1.size(), l1_error.size());
	for (int i = 0; i < 10; i++)
		std::cout << l1_out_error[i] << " ";
	std::cout << std::endl;
	for (int i = 0; i < 10; i++)
		std::cout << l1[i] << " ";
	std::cout << std::endl;
	for (int i = 0; i < 10; i++)
		std::cout << l1_sigmoid_prime[i] << " ";
	std::cout << std::endl;
	for (int i = 0; i < 10; i++)
		std::cout << l1_error[i] << " ";
	std::cout << std::endl;
//	for (int i = 0; i < 500; i++)
//		std::cout << i << ": " << l2[i] << ", " << l2_sigmoid_prime[i] << std::endl;
	*/
}


double neural_net::get_error()
{
	int n = labels.size();
	int blockSize = 256;
	int numBlocks = (n + blockSize - 1) / blockSize;
	
	thrust::device_vector<double> error_sums(labels.size(), 0);
	cuda_get_error<<<numBlocks, blockSize>>>(n,
			thrust::raw_pointer_cast(l3.data()), 
			thrust::raw_pointer_cast(labels.data()), 
			thrust::raw_pointer_cast(error_sums.data()), 
			opts.n_o);
	
	cudaDeviceSynchronize(); 
	return thrust::reduce(error_sums.begin(), error_sums.end(), 0.0, thrust::plus<double>());
}


thrust::device_vector<float> neural_net::get_error_gradient()
{
	int n = l3.size();
	int blockSize = 256;
	int numBlocks = (n + blockSize - 1) / blockSize;
	
	thrust::device_vector<float> error_gradient(l3.size(), 0);
	cuda_get_error_gradient<<<numBlocks, blockSize>>>(n,
			thrust::raw_pointer_cast(l3.data()), 
			thrust::raw_pointer_cast(labels.data()), 
			thrust::raw_pointer_cast(error_gradient.data()), 
			opts.n_o);
	
	cudaDeviceSynchronize(); 
	return error_gradient;
}


double neural_net::get_accuracy()
{
	double acc = 0;
	for (int i = 0; i < inputs.size() / opts.n_x; i++)
	{
		thrust::device_vector<float>::iterator iter = thrust::max_element(l3.begin() + (i * opts.n_o), 
					l3.begin() + ((i + 1) * opts.n_o));
		unsigned int pos = iter - (l3.begin() + (i * opts.n_o));
		if (pos == labels[i])
			acc++;
	}

	return acc / (inputs.size() / opts.n_x);
}




thrust::device_vector<float> neural_net::matrix_multiply(thrust::device_vector<float> A, thrust::device_vector<float> B, int a_rows, int a_cols, int b_rows, int b_cols, int op)
{

	int n = a_rows, m = a_cols, r = b_cols;
	float alpha = 1.0f, beta = 0.0f;
	thrust::device_vector<float> new_result;

	//AxB
	if (op == 0x01)
	{
		thrust::device_vector<float> result(n*r, 0);
		new_result = thrust::device_vector<float>(n*r, 0);

		//Multiply
		cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, n, r, m, &alpha, thrust::raw_pointer_cast(A.data()), m, thrust::raw_pointer_cast(B.data()), r, &beta, thrust::raw_pointer_cast(result.data()), n);
		cudaDeviceSynchronize();

		//Transpose result to row major
		cublasSgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, r, n, &alpha, thrust::raw_pointer_cast(result.data()), n, &beta, thrust::raw_pointer_cast(result.data()), r, thrust::raw_pointer_cast(new_result.data()), r);
		cudaDeviceSynchronize(); 
	}
	else if (op == 0x02)
	{
		thrust::device_vector<float> transpose(n*m, 0);
		thrust::device_vector<float> result(m*r, 0);
		new_result = thrust::device_vector<float> (m*r, 0);

		//Transpose
		cublasSgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha, thrust::raw_pointer_cast(A.data()), m, &beta, thrust::raw_pointer_cast(A.data()), n, thrust::raw_pointer_cast(transpose.data()), n);
		cudaDeviceSynchronize(); 
		//Multiply
		cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, m, r, n, &alpha, thrust::raw_pointer_cast(transpose.data()), n, thrust::raw_pointer_cast(B.data()), r, &beta, thrust::raw_pointer_cast(result.data()), m);
		cudaDeviceSynchronize();
		//Transpose result to row major
		cublasSgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, r, m, &alpha, thrust::raw_pointer_cast(result.data()), m, &beta, thrust::raw_pointer_cast(result.data()), r, thrust::raw_pointer_cast(new_result.data()), r);
		cudaDeviceSynchronize(); 
	}
	
	else if (op == 0x03)
	{

		r = b_rows;
		thrust::device_vector<float> transpose(m*r, 0);
		thrust::device_vector<float> result(n*r, 0);
		new_result = thrust::device_vector<float> (n*r, 0);

		//Transpose
		cublasSgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, r, m, &alpha, thrust::raw_pointer_cast(B.data()), m, &beta, thrust::raw_pointer_cast(B.data()), r, thrust::raw_pointer_cast(transpose.data()), r);
		cudaDeviceSynchronize(); 
		//Multiply
		cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, n, r, m, &alpha, thrust::raw_pointer_cast(A.data()), m, thrust::raw_pointer_cast(transpose.data()), r, &beta, thrust::raw_pointer_cast(result.data()), n);
		cudaDeviceSynchronize();
		//Transpose result to row major
		cublasSgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, r, n, &alpha, thrust::raw_pointer_cast(result.data()), n, &beta, thrust::raw_pointer_cast(result.data()), r, thrust::raw_pointer_cast(new_result.data()), r);
		cudaDeviceSynchronize(); 
	}
	
	return new_result;
}

