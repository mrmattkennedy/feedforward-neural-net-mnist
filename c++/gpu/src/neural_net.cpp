#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <algorithm>
#include <chrono>
#include <numeric>

#include "neural_net.hpp"
#include "data_reader.hpp"
#include "options.hpp"


struct RandGen
{
	//N is size of vector, M is iteration
	unsigned int M;
	unsigned int N;

	RandGen(unsigned int _M=1, unsigned int _N=1) : M(_M), N(_N) {};

	__device__ float operator () (unsigned int thread_id)
	{
		thrust::default_random_engine randEng;
		randEng.discard(N * (M+1) * thread_id);
		thrust::uniform_real_distribution<float> uniDist(-1.0, 1.0);
		return uniDist(randEng);
	}
};


struct dp
{
	float *A, *B;
	int m,n,r;
	dp(float *_A, float *_B, int _m, int _n, int _r): A(_A), B(_B), m(_m), n(_n), r(_r) {};

	__host__ __device__
	float operator()(size_t idx){
		float sum = 0.0f;
		int row = idx/r;
		int col = idx - (row*r); // cheaper modulo

		for (int i = 0; i < m; i++)
			sum += A[col + row*i] * B[col + row*i];
		return sum;
	}
};


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
	start = clock();
	create_arch();
	int train_size = 60000;

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

	v_w1 = init_velocity(opts.n_x, opts.n_h1);
	v_b1 = init_velocity(1, opts.n_h1);
	v_w2 = init_velocity(opts.n_h1, opts.n_h2);
	v_b2 = init_velocity(1, opts.n_h2);
	v_w3 = init_velocity(opts.n_h2, opts.n_o);
	v_b3 = init_velocity(1, opts.n_o);

	std::cout << v_w1[0][0] << std::endl;
}


std::vector<thrust::device_vector<float>> neural_net::init_weight(int insize, int outsize)
{

	std::vector<thrust::device_vector<float>> d_vec(insize, thrust::device_vector<float>(outsize));
	thrust::device_vector<float> temp(outsize);

	for (unsigned int i = 0; i < insize; i++)
	{
		thrust::transform(
			thrust::counting_iterator<int>(0),
			thrust::counting_iterator<int>(outsize),
			temp.begin(),
			RandGen(i, outsize));
		d_vec[i] = temp;
	}
	temp.clear();
	return d_vec;
}


std::vector<thrust::device_vector<float>> neural_net::init_velocity(int insize, int outsize)
{
	std::vector<thrust::device_vector<float>> d_vec(insize, thrust::device_vector<float>(outsize));

	for (unsigned int i = 0; i < insize; i++)
		d_vec[i] = thrust::device_vector<float>(outsize, 0);

	return d_vec;
}

void neural_net::feed_forward()
{
}

void neural_net::back_propagation()
{

}

int neural_net::get_error()
{
	return 0;
}


double neural_net::get_accuracy()
{
	return 0.0;
}
