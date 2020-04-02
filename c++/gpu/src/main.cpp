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


int main(void)
{
	std::vector<thrust::device_vector<float>> d_vec(700, thrust::device_vector<float>(500));
	thrust::device_vector<float> temp(500);
	for (unsigned int i = 0; i < 700; i++)
	{
		thrust::counting_iterator<unsigned int> index_sequence_begin(i);
		thrust::transform(
			thrust::counting_iterator<int>(0),
			thrust::counting_iterator<int>(500),
			temp.begin(),
			RandGen(i, 500));

		d_vec[i] = temp;
	}
	temp.clear();
	return 0;
}
