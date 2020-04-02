#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/generate.h>
#include <thrust/random.h>

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
	create_arch();
	int train_size = 60000;

	std::vector<int> shuffle_vector(train_size);
	std::iota(shuffle_vector.begin(), shuffle_vector.end(), 0);

	clock_t start, end;
	start = clock();

	for (int i = 0; i < opts.epochs; i++)
	{	
		std::random_shuffle(shuffle_vector.begin(), shuffle_vector.end());
		opts.alpha *= (1 / (1 + opts.decay * i));

		for (int j = 0; j < opts.batches; j++)
		{
		}

	}
	end = clock();
	double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
	printf("%f\n", time_taken);
		
}

void neural_net::create_arch()
{
	thrust::device_vector<float> d_vec(100000);
	thrust::transform(
		thrust::make_counting_iterator(0),
		thrust::make_counting_iterator(100),
		d_vec.begin(),
		RandGen());
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
