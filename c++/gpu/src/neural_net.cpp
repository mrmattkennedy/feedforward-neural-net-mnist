#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
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
#include <tuple>
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
	unsigned int N, seed;

	__host__ __device__
	RandGen(unsigned int _N, unsigned int _seed) : N(_N), seed(_seed) {};

	
	__host__ __device__
	float operator () (unsigned int thread_id)
	{
		thrust::minstd_rand rng;
		rng.seed(seed);
		rng.discard(N * thread_id);
		thrust::random::normal_distribution<float> dist(0, 1);
		return dist(rng);
	}
};

template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T>
{
	T C; // number of columns

	__host__ __device__
	linear_index_to_row_index(T C) : C(C) {}

	__host__ __device__
	T operator()(T i)
	{
		return i / C;
	}
};

typedef thrust::tuple<int,float> argMaxType;
struct argMax : public thrust::binary_function<argMaxType,argMaxType,argMaxType>
{
	__host__ __device__
	argMaxType operator()(const argMaxType& a, const argMaxType& b) const
	{
		if (thrust::get<1>(a) > thrust::get<1>(b)){
			return a;
		} else {
			return b;
		}
	}
};

struct GetIndices
{
	unsigned int N;
	__host__ __device__
	GetIndices(unsigned int N) : N(N) {}

	__host__ __device__
	int operator()(argMaxType t)
	{
		return thrust::get<0>(t) % N;
	}
};


__global__ 
void cuda_shuffle_x(int n, int *shuffled_idx, float *inputs, float *shuffled_x, int n_x)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		shuffled_x[i] = inputs[(shuffled_idx[i / n_x] * n_x) + (i % n_x)];
}

__global__ 
void cuda_shuffle_y(int n, int *shuffled_idx, float *labels, float *shuffled_y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		shuffled_y[i] = labels[shuffled_idx[i]];
}

__global__ 
void cuda_add_bias(int n, float *x, float *bias, int n_x)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		x[i] += bias[i % n_x];
}

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
void cuda_get_error_gradient(int n, float *labels, float *gradient, int n_o)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		if (i % n_o == labels[i / n_o])
			gradient[i] -= 1;
}

__global__ 
void cuda_get_bias_delta(int n, float *gradient, float *layer_bias, int num_nodes)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		atomicAdd(&layer_bias[i % num_nodes], gradient[i]);
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

__global__ 
void cuda_get_accuracy(int n, float *outputs, float *labels, int *accuracy)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		if (outputs[i] == labels[i])
			accuracy[i] = 1;
}

__global__ 
void cuda_get_conf(int n, float *outputs, float *labels, int *conf, int n_o)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		atomicAdd(&conf[((int)outputs[i] * n_o) + (int)labels[i]], 1);
}

neural_net::neural_net(std::string path) : data(path) 
{
	//empty	
}

neural_net::~neural_net()
{
	//empty
}

void neural_net::reset()
{
	inputs = data.m_images;
	labels = data.m_labels;
	test_in = thrust::device_vector<float>(test_size*opts.n_x);
	test_labels = thrust::device_vector<float>(test_size);
	shuffled_x = thrust::device_vector<float>(inputs.size(), 1);
	shuffled_y = thrust::device_vector<float>(labels.size(), 1);
	batch_x = thrust::device_vector<float>(opts.batch_size*opts.n_x);
	batch_y = thrust::device_vector<float>(opts.batch_size);
	opts.alpha = 0.002;
}
void neural_net::shuffle()
{
	//Initialize shuffled vector of ints with range 0 to size of data
	int data_size = train_size + test_size;

	std::vector<int> shuffle_vec(data_size);
	std::iota(std::begin(shuffle_vec), std::end(shuffle_vec), 0);
	srand(unsigned(time(NULL)));
	std::random_shuffle(shuffle_vec.begin(), shuffle_vec.end());
	
	//Iterate and assign new shuffled data using cuda
	thrust::device_vector<int> shuffle_idx = shuffle_vec;
	
	int numBlocks = (shuffled_x.size() + blockSize - 1) / blockSize;
	cuda_shuffle_x<<<numBlocks, blockSize>>>(shuffled_x.size(), 
			thrust::raw_pointer_cast(shuffle_idx.data()),
			thrust::raw_pointer_cast(inputs.data()),
			thrust::raw_pointer_cast(shuffled_x.data()),
			opts.n_x);
	cudaDeviceSynchronize(); 

	numBlocks = (shuffled_y.size() + blockSize - 1) / blockSize;
	cuda_shuffle_y<<<numBlocks, blockSize>>>(shuffled_y.size(), 
			thrust::raw_pointer_cast(shuffle_idx.data()),
			thrust::raw_pointer_cast(labels.data()),
			thrust::raw_pointer_cast(shuffled_y.data()));
	cudaDeviceSynchronize(); 

	//Assign new data
	thrust::copy(shuffled_x.begin(), shuffled_x.begin() + (train_size*opts.n_x), inputs.begin());
	thrust::copy(shuffled_y.begin(), shuffled_y.begin() + train_size, labels.begin());

	thrust::copy(shuffled_x.begin() + (train_size*opts.n_x), shuffled_x.end(), test_in.begin());
	thrust::copy(shuffled_y.begin() + train_size, shuffled_y.end(), test_labels.begin());	
}

std::tuple<std::tuple<int, __int64>, std::vector<std::tuple<int, float>>> neural_net::train(int batch_size)
{
	if (batch_size != 0)
	{
		opts.batch_size = batch_size;
		opts.batches = train_size / batch_size;
	}
	printf("Batch size: %d\tBatches: %d\n", opts.batch_size, opts.batches);
	reset();
	create_arch();
	cublasCreate(&h);
	std::vector<std::tuple<int, float>> ret;
	auto start = std::chrono::high_resolution_clock::now();		
	int numBlocks = (opts.n_o * opts.n_h2 + blockSize - 1) / blockSize;

	for (int i = 0; i < opts.epochs; i++)
	{
		shuffle();
		/*
		for (int i = 0; i < 784; i++)
		{
			if (i % 28 == 0)
				std::cout << std::endl;
			
			//std::cout << inputs[i] << " ";
			printf("%3.f", (float)inputs[i]);
		}
		std::cout << std::endl;
		*/
		
		opts.alpha *= (1 / (1 + opts.decay * i));
		for (int j = 0; j < opts.batches; j++)
		{
			thrust::copy(inputs.begin() + (j * (opts.batch_size*opts.n_x)), inputs.begin() + ((j+1) * (opts.batch_size*opts.n_x)), batch_x.begin());
			thrust::copy(labels.begin() + (j * opts.batch_size), labels.begin() + ((j+1) * opts.batch_size), batch_y.begin());
			
			feed_forward(batch_x);
			back_propagation();

			//Update velocities
			numBlocks = (opts.n_o * opts.n_h2 + blockSize - 1) / blockSize;
			cuda_update_velocity<<<numBlocks, blockSize>>>(opts.n_o * opts.n_h2, 
					thrust::raw_pointer_cast(v_w3.data()),
					thrust::raw_pointer_cast(l3_delta.data()),
					opts.beta);
			cudaDeviceSynchronize(); 

			numBlocks = (opts.n_h2 * opts.n_h1 + blockSize - 1) / blockSize;
			cuda_update_velocity<<<numBlocks, blockSize>>>(opts.n_h2 * opts.n_h1, 
					thrust::raw_pointer_cast(v_w2.data()),
					thrust::raw_pointer_cast(l2_delta.data()),
					opts.beta);
			cudaDeviceSynchronize(); 

			numBlocks = (opts.n_h1 * opts.n_x + blockSize - 1) / blockSize;
			cuda_update_velocity<<<numBlocks, blockSize>>>(opts.n_h1 * opts.n_x,
					thrust::raw_pointer_cast(v_w1.data()),
					thrust::raw_pointer_cast(l1_delta.data()),
					opts.beta);
			cudaDeviceSynchronize(); 

			//Update weights
			numBlocks = (opts.n_o * opts.n_h2 + blockSize - 1) / blockSize;
			cuda_update_weight<<<numBlocks, blockSize>>>(opts.n_o * opts.n_h2, 
					thrust::raw_pointer_cast(w3.data()),
					thrust::raw_pointer_cast(v_w3.data()),
					opts.alpha);
			cudaDeviceSynchronize(); 

			numBlocks = (opts.n_h2 * opts.n_h1 + blockSize - 1) / blockSize;
			cuda_update_weight<<<numBlocks, blockSize>>>(opts.n_h2 * opts.n_h1, 
					thrust::raw_pointer_cast(w2.data()),
					thrust::raw_pointer_cast(v_w2.data()),
					opts.alpha);
			cudaDeviceSynchronize(); 

			numBlocks = (opts.n_h1 * opts.n_x + blockSize - 1) / blockSize;
			cuda_update_weight<<<numBlocks, blockSize>>>(opts.n_h1 * opts.n_x, 
					thrust::raw_pointer_cast(w1.data()),
					thrust::raw_pointer_cast(v_w1.data()),
					opts.alpha);
			cudaDeviceSynchronize(); 
		}
		
		if (i % 1 == 0)
		{
			//Feed forward test set
			feed_forward(test_in);
			float acc = get_accuracy();
			printf("%d:\tError:%f\tTest acc:%f\n", i, model_error, acc);
			ret.push_back(std::tuple<int, float>(opts.batch_size, acc));
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
	std::tuple<int, __int64> timing(opts.batch_size, diff);
	std::cout << "Took " << diff << " milliseconds\n";
	cublasDestroy(h);
	return std::tuple<std::tuple<int, __int64>, std::vector<std::tuple<int, float>>>(timing, ret);
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

	thrust::device_vector<float> w(insize * outsize);
	thrust::transform(
		thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(insize * outsize),
		w.begin(),
		RandGen(insize * outsize, (unsigned(time(NULL)))));

	float factor = sqrt(1.0 / insize);
	thrust::for_each(w.begin(), w.end(), thrust::placeholders::_1 *= factor);
	return w;
}



void neural_net::feed_forward(thrust::device_vector<float> in)
{

	//Using thrust transform with struct doesn't work for large vectors. Need to use cublas GEMM (general matrix multiply) algorithms.
//	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n*r), result.begin(), dp(thrust::raw_pointer_cast(inputs.data()), thrust::raw_pointer_cast(w1.data()), m, n, r));
	
	//Dot product of inputs and w1
	int n = in.size() / opts.n_x, m = opts.n_x, r = opts.n_h1; //inputs is 70000x784 (NxM), w1 is 784x600 (MxR),
	auto a1 = matrix_multiply(in, w1, n, m, m, r, NORMAL);
	cudaDeviceSynchronize(); 
	
	//Add bias
	int numBlocks = (n*r + blockSize - 1) / blockSize;
	cuda_add_bias<<<numBlocks, blockSize>>>(n*r, thrust::raw_pointer_cast(a1.data()), thrust::raw_pointer_cast(b1.data()), r);
	cudaDeviceSynchronize();

	//clip a1 values, assign to l1
	l1 = clip(a1);

	//Cuda kernel for __expf, fast exponent
	sigmoid_cuda<<<numBlocks, blockSize>>>(n*r, thrust::raw_pointer_cast(l1.data()));
	cudaDeviceSynchronize();
	
	
	//Hidden layer 2
	m = opts.n_h1, r = opts.n_h2;
	auto a2 = matrix_multiply(l1, w2, n, m, m, r, NORMAL);
	cudaDeviceSynchronize(); 
	
	//Add bias
	numBlocks = (n*r + blockSize - 1) / blockSize;
	cuda_add_bias<<<numBlocks, blockSize>>>(n*r, thrust::raw_pointer_cast(a2.data()), thrust::raw_pointer_cast(b2.data()), r);
	cudaDeviceSynchronize();

	//clip a2 values, assign to l2
	l2 = clip(a2);

	//Cuda kernel for __expf, fast exponent
	sigmoid_cuda<<<numBlocks, blockSize>>>(n*r, thrust::raw_pointer_cast(l2.data()));
	cudaDeviceSynchronize();


	//Output layer
	m = opts.n_h2, r = opts.n_o;
	auto a3 = matrix_multiply(l2, w3, n, m, m, r, NORMAL);
	cudaDeviceSynchronize(); 
	
	//Add bias
	numBlocks = (n*r + blockSize - 1) / blockSize;
	cuda_add_bias<<<numBlocks, blockSize>>>(n*r, thrust::raw_pointer_cast(a3.data()), thrust::raw_pointer_cast(b3.data()), r);

	//Get exponent
	l3 = a3;
	numBlocks = (n*r + blockSize - 1) / blockSize;
	cuda_get_exp<<<numBlocks, blockSize>>>(n*r, thrust::raw_pointer_cast(l3.data()));
	cudaDeviceSynchronize(); 

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
	thrust::for_each(in.begin(), in.end(), thrust::placeholders::_1 /= factor);
	return in;
}

void neural_net::back_propagation()
{	
	model_error = get_error();
	auto error_gradient = get_error_gradient();
	int n = batch_x.size() / opts.n_x, m = opts.n_h2, r = opts.n_o;
	
	//Output layer
	//Get out delta
	l3_delta = matrix_multiply(l2, error_gradient, n, m, n, r, TRANSPOSE_A);
	thrust::for_each(l3_delta.begin(), l3_delta.end(), thrust::placeholders::_1 /= n);
	cudaDeviceSynchronize(); 

	//Get bias delta
	int numBlocks = (n*r + blockSize - 1) / blockSize;
	l3_bias_delta = thrust::device_vector<float>(r, 0);
	cuda_get_bias_delta<<<numBlocks, blockSize>>>(n*r,
			thrust::raw_pointer_cast(error_gradient.data()), 
			thrust::raw_pointer_cast(l3_bias_delta.data()), 
			r);
	cudaDeviceSynchronize(); 
	//Divide after - significantly less divide operations this way
	thrust::for_each(l3_bias_delta.begin(), l3_bias_delta.end(), thrust::placeholders::_1 /= n);
	cudaDeviceSynchronize(); 


	//Hidden layer 2	
	//Get hidden layer 2 output error
	m = opts.n_o, r = opts.n_h2;
	auto l2_out_error = matrix_multiply(error_gradient, w3, n, m, r, m, TRANSPOSE_B);

	//Get sigmoid derivative for hidden layer 2
	thrust::device_vector<float> l2_sigmoid_prime(n*r, 0);
	numBlocks = (n*r + blockSize - 1) / blockSize;
	sigmoid_prime_cuda<<<numBlocks, blockSize>>>(n*r,
			thrust::raw_pointer_cast(l2.data()), 
			thrust::raw_pointer_cast(l2_sigmoid_prime.data()));
	cudaDeviceSynchronize(); 

	//Get hidden layer 2 error
	thrust::device_vector<float> l2_error(n*r, 0);
	cuda_get_layer_error<<<numBlocks, blockSize>>>(n*r,
			thrust::raw_pointer_cast(l2_out_error.data()), 
			thrust::raw_pointer_cast(l2.data()), 
			thrust::raw_pointer_cast(l2_sigmoid_prime.data()),
			thrust::raw_pointer_cast(l2_error.data()));
	cudaDeviceSynchronize(); 
	
	//Get hidden layer 2 deltas
	m = opts.n_h1, r = opts.n_h2;
	l2_delta = matrix_multiply(l1, l2_error, n, m, n, r, TRANSPOSE_A);
	
	//Get bias delta
	l2_bias_delta = thrust::device_vector<float>(r, 0);
	cuda_get_bias_delta<<<numBlocks, blockSize>>>(n*r,
			thrust::raw_pointer_cast(l2_error.data()), 
			thrust::raw_pointer_cast(l2_bias_delta.data()), 
			r);
	cudaDeviceSynchronize(); 
	//Divide after - significantly less divide operations this way
	thrust::for_each(l2_bias_delta.begin(), l2_bias_delta.end(), thrust::placeholders::_1 /= n);
	cudaDeviceSynchronize(); 
	

	//Hidden layer 1
	//Get hidden layer 1 output error
	auto l1_out_error = matrix_multiply(l2_error, w2, n, r, m, r, TRANSPOSE_B);
	
	//Get sigmoid derivative for hidden layer 1
	thrust::device_vector<float> l1_sigmoid_prime(n*m, 0);
	numBlocks = (n*m + blockSize - 1) / blockSize;
	sigmoid_prime_cuda<<<numBlocks, blockSize>>>(n*m,
			thrust::raw_pointer_cast(l1.data()), 
			thrust::raw_pointer_cast(l1_sigmoid_prime.data()));
	cudaDeviceSynchronize(); 
	
	//Get hidden layer 1 error
	m = opts.n_h2, r = opts.n_h1;
	thrust::device_vector<float> l1_error(n*r, 0);
	numBlocks = (n*r + blockSize - 1) / blockSize;
	cuda_get_layer_error<<<numBlocks, blockSize>>>(n*r,
			thrust::raw_pointer_cast(l1_out_error.data()), 
			thrust::raw_pointer_cast(l1.data()), 
			thrust::raw_pointer_cast(l1_sigmoid_prime.data()),
			thrust::raw_pointer_cast(l1_error.data()));
	cudaDeviceSynchronize(); 

	//Get hidden layer 1 deltas
	m = opts.n_x, r = opts.n_h1;
	l1_delta = matrix_multiply(inputs, l1_error, n, m, n, r, TRANSPOSE_A);
	
	//Get bias delta
	l1_bias_delta = thrust::device_vector<float>(r, 0);
	numBlocks = (n*r + blockSize - 1) / blockSize;
	cuda_get_bias_delta<<<numBlocks, blockSize>>>(n*r,
			thrust::raw_pointer_cast(l1_error.data()), 
			thrust::raw_pointer_cast(l1_bias_delta.data()), 
			r);
	cudaDeviceSynchronize(); 
	//Divide after - significantly less divide operations this way
	thrust::for_each(l1_bias_delta.begin(), l1_bias_delta.end(), thrust::placeholders::_1 /= n);
	cudaDeviceSynchronize(); 	
}

double neural_net::get_error()
{
	int n = batch_y.size();
	int numBlocks = (n + blockSize - 1) / blockSize;
	thrust::device_vector<double> error_sums(n, 0);
	cuda_get_error<<<numBlocks, blockSize>>>(n,
			thrust::raw_pointer_cast(l3.data()), 
			thrust::raw_pointer_cast(batch_y.data()), 
			thrust::raw_pointer_cast(error_sums.data()), 
			opts.n_o);
	cudaDeviceSynchronize();
	return thrust::reduce(error_sums.begin(), error_sums.end(), 0.0, thrust::plus<double>());
}


thrust::device_vector<float> neural_net::get_error_gradient()
{
	int n = l3.size();
	int numBlocks = (n + blockSize - 1) / blockSize;
	auto error_gradient = l3;
	cuda_get_error_gradient<<<numBlocks, blockSize>>>(n,
			thrust::raw_pointer_cast(batch_y.data()), 
			thrust::raw_pointer_cast(error_gradient.data()), 
			opts.n_o);
	
	cudaDeviceSynchronize(); 
	return error_gradient;
}

double neural_net::get_accuracy()
{
	//Get sizes for indexing
	int nRows = l3.size() / opts.n_o;
	int nColumns = opts.n_o;
	
	//Allocate vectors for vector of tuples and counting index for each row
	thrust::device_vector<argMaxType> row_argmaxes(nRows);
	thrust::device_vector<int> row_indices(nRows);
	
	//Get vector of tuples of max and max position for each row
	thrust::reduce_by_key
		(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(nColumns)),
		thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(nColumns)) + (nRows*nColumns),
		thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0),l3.begin())),
		row_indices.begin(),
		row_argmaxes.begin(),
		thrust::equal_to<int>(),
		argMax());
	
	//Get just the indices from the vector of max tuples for each row
	thrust::device_vector<float> indices(nRows);
	thrust::transform(row_argmaxes.begin(), row_argmaxes.end(), indices.begin(), GetIndices(opts.n_o));

	//Get accuracy		
	int numBlocks = (nRows + blockSize - 1) / blockSize;
	
	thrust::device_vector<int> accuracies(test_labels.size(), 0);
	cuda_get_accuracy<<<numBlocks, blockSize>>>(nRows,
			thrust::raw_pointer_cast(indices.data()), 
			thrust::raw_pointer_cast(test_labels.data()), 
			thrust::raw_pointer_cast(accuracies.data())); 
	cudaDeviceSynchronize(); 
	
	/*
	//Confusion matrix
	thrust::device_vector<int> conf_mat(opts.n_o * opts.n_o, 0);
	cuda_get_conf<<<numBlocks, blockSize>>>(nRows,
			thrust::raw_pointer_cast(indices.data()), 
			thrust::raw_pointer_cast(test_labels.data()), 
			thrust::raw_pointer_cast(conf_mat.data()),
			opts.n_o); 
	cudaDeviceSynchronize(); 
	printf("\nTest confusion matrix:\n");
	printf("      ");
	for (int i = 0; i < 10; i++)
		printf("%4d ", i);
	std::cout << std::endl;
	for (int i = 0; i < conf_mat.size(); i++)
	{
		if (i % opts.n_o == 0)
		{
			std::cout << std::endl;
			printf("%4d  ", i / opts.n_o);
		}
		//std::cout << conf_mat[i] << " ";
		printf("%4d ", (int)conf_mat[i]);
	}
	std::cout << std::endl;
	*/
	//Reduce and divide
	float total = thrust::reduce(thrust::device, accuracies.begin(), accuracies.end());
	return total / nRows;
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

