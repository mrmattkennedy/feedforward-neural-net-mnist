#include "neural_net.hpp"
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
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


int main(int argc, char** argv)
{
	std::string base_path = "..\\..\\MNIST data\\";
	int n = 1000, m = 1000, r = 1000;
	thrust::device_vector<float> vec1(m*n);
	std::generate(vec1.begin(), vec1.end(), rand);

	thrust::device_vector<float> vec2(n*r);
	std::generate(vec2.begin(), vec2.end(), rand);
	
	thrust::device_vector<float> results(n*r, 0);
	clock_t start, end;
	start = clock();

	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n*r), results.begin(), dp(thrust::raw_pointer_cast(vec1.data()), thrust::raw_pointer_cast(vec2.data()), m, n, r));

	end = clock();
	double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
	printf("%f\n", time_taken);
	//neural_net nn(base_path);

	//nn.train();
	
	return 0;
}
