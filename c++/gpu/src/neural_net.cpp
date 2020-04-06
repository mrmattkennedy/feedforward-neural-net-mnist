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
		thrust::uniform_real_distribution<float> uniDist(0.01, 1.0);
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
void cuda_get_error(int n, float *outputs, float *labels, float *sums, int n_o)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		sums[i] = -__logf(outputs[(i*n_o) + (int)labels[i]] + 1e-10);
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

	start = clock();
	//std::vector<int> shuffle_vector(train_size);
	//std::iota(shuffle_vector.begin(), shuffle_vector.end(), 0);

	
	/*
	for (int i = 0; i < opts.epochs; i++)
	{	
		std::random_shuffle(shuffle_vector.begin(), shuffle_vector.end());
		opts.alpha *= (1 / (1 + opts.decay * i));

		for (int j = 0; j < opts.batches; j++)
		{
		}

	}
	*/
	feed_forward();
	back_propagation();
	end = clock();
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
	cublasHandle_t h;
	cublasCreate(&h);
	float alpha = 1.0f, beta=0.0f;
	int blockSize = 256;
	
	//dot product of inputs and w1
	int n = inputs.size() / opts.n_x, m = opts.n_x, r = opts.n_h1; //inputs is 70000x784 (NxM), w1 is 784x600 (MxR),
	thrust::device_vector<float> a1(n*r, 0);
	cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, r, m, &alpha, thrust::raw_pointer_cast(inputs.data()), n, thrust::raw_pointer_cast(w1.data()), m, &beta, thrust::raw_pointer_cast(a1.data()), n);
	cudaDeviceSynchronize(); 
	
	//clip a1 values, assign to l1
	l1 = clip(a1);

	//Cuda kernel for __expf, fast exponent
	int numBlocks = (n*r + blockSize - 1) / blockSize;
	sigmoid_cuda<<<numBlocks, blockSize>>>(n*r, thrust::raw_pointer_cast(l1.data()));
	cudaDeviceSynchronize();
	
	
	//Hidden layer 2
	m = opts.n_h1, r = opts.n_h2;
	thrust::device_vector<float> a2(n*r, 0);

	cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, r, m, &alpha, thrust::raw_pointer_cast(l1.data()), n, thrust::raw_pointer_cast(w2.data()), m, &beta, thrust::raw_pointer_cast(a2.data()), n);
	cudaDeviceSynchronize(); 
	
	//clip a2 values, assign to l2
	l2 = clip(a2);

	//Cuda kernel for __expf, fast exponent
	numBlocks = (n*r + blockSize - 1) / blockSize;
	sigmoid_cuda<<<numBlocks, blockSize>>>(n*r, thrust::raw_pointer_cast(l2.data()));
	cudaDeviceSynchronize();


	//Output layer
	m = opts.n_h2, r = opts.n_o;
	thrust::device_vector<float> a3(n*r, 0);
	cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, r, m, &alpha, thrust::raw_pointer_cast(l2.data()), n, thrust::raw_pointer_cast(w3.data()), m, &beta, thrust::raw_pointer_cast(a3.data()), n);
	cudaDeviceSynchronize(); 
	
	//Get exponent
	l3 = clip(a3, 50);

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
	numBlocks = (n*r + blockSize - 1) / blockSize;
	softmax_cuda<<<numBlocks, blockSize>>>(n*r, thrust::raw_pointer_cast(l3.data()), thrust::raw_pointer_cast(exp_sums.data()), opts.n_o);

	cudaDeviceSynchronize(); 
	cublasDestroy(h);
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

	float alpha = 1.0f, beta=0.0f;
	int blockSize = 256;	
	int n = inputs.size() / opts.n_x, m = opts.n_h2, r = opts.n_o;

	thrust::device_vector<float> l2_transpose(l2.size());
	thrust::device_vector<float> out_delta(m*r, 0);

	cublasHandle_t h;
	cublasCreate(&h);

	cublasSgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha, thrust::raw_pointer_cast(l2.data()), m, &beta, thrust::raw_pointer_cast(l2.data()), n, thrust::raw_pointer_cast(l2_transpose.data()), n);
	cudaDeviceSynchronize(); 
	cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, m, m, r, &alpha, thrust::raw_pointer_cast(l2_transpose.data()), n, thrust::raw_pointer_cast(error_gradient.data()), m, &beta, thrust::raw_pointer_cast(out_delta.data()), m);
	cudaDeviceSynchronize(); 

	thrust::copy_n(out_delta.begin(), 10, std::ostream_iterator<float>(std::cout, ","));
	cublasDestroy(h);
}

double neural_net::get_error()
{
	int n = labels.size();
	int blockSize = 256;
	int numBlocks = (n + blockSize - 1) / blockSize;
	
	thrust::device_vector<float> error_sums(labels.size(), 0);
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
	return 0.0;
}
