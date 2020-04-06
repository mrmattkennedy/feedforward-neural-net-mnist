#include "neural_net.hpp"

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




int main(int argc, char** argv)
{
	std::string base_path = "..\\..\\MNIST data\\";
	neural_net nn(base_path);
	//nn.train();


	int n = 3, m = 2, r = 3;
	thrust::device_vector<float> v1(n*m, 1);
	thrust::device_vector<float> v2(m*r, 1);
	v1[0] = 1;
	v1[1] = 1;
	v1[2] = 2;
	v1[3] = 2;
	v1[4] = 3;
	v1[5] = 3;
	v2[0] = 1;
	v2[1] = 1;
	v2[2] = 2;
	v2[3] = 2;
	v2[4] = 3;
	v2[5] = 3;
	thrust::device_vector<float> result(n*r, 0);
	thrust::device_vector<float> transpose(n*m, 0);
	cublasHandle_t h;
	cublasCreate(&h);
	float alpha = 1.0f, beta=0.0f;
	
	//cublasScopy(h, n*m, thrust::raw_pointer_cast(v1.data()), m, thrust::raw_pointer_cast(result.data()), 1);
	//for (int i = 0; i < v1.size(); i++)
//		result[i] = v1[((i * m) % (n * m))];
	
	cublasSgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha, thrust::raw_pointer_cast(v1.data()), m, &beta, thrust::raw_pointer_cast(v1.data()), n, thrust::raw_pointer_cast(transpose.data()), n);
	cudaDeviceSynchronize(); 
	
	for (int i = 0; i < v1.size(); i++)
		std::cout << v1[i] << " ";
	std::cout << std::endl;

	for (int i = 0; i < transpose.size(); i++)
		std::cout << transpose[i] << " ";
	std::cout << std::endl;

	//cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, r, n, m, &alpha, thrust::raw_pointer_cast(v1.data()), m, thrust::raw_pointer_cast(transpose.data()), n, &beta, thrust::raw_pointer_cast(result.data()), n);
	//cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, r, n, m, &alpha, thrust::raw_pointer_cast(v1.data()), n, thrust::raw_pointer_cast(v2.data()), m, &beta, thrust::raw_pointer_cast(result.data()), r);
	
	cudaDeviceSynchronize(); 
	cublasDestroy(h);
	printf("Size is %d\n", result.size());
	for (int i = 0; i < result.size(); i++)
	{
		std::cout << result[i] << " ";
	}
	return 0;
}
