//#include "neural_net.hpp"
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
	//neural_net nn(base_path);
	//nn.train();
	int n = 4, m = 3, r = 2;
	thrust::device_vector<float> v1(n*m, 1);
	thrust::device_vector<float> v2(n*r, 1);
	v2[0] = 13;
	v2[1] = 14;
	v2[2] = 15;
	v2[3] = 16;
	v2[4] = 17;
	v2[5] = 18;
	v2[6] = 19;
	v2[7] = 20;
	v1[0] = 1;
	v1[1] = 5;
	v1[2] = 9;
	v1[3] = 2;
	v1[4] = 6;
	v1[5] = 10;
	v1[6] = 3;
	v1[7] = 7;
	v1[8] = 11;
	v1[9] = 4;
	v1[10] = 8;
	v1[11] = 12;
	thrust::device_vector<float> result(m*r, 0);
	thrust::device_vector<float> transpose(n*m, 0);
	cublasHandle_t h;
	cublasCreate(&h);
	float alpha = 1.0f, beta=0.0f;
	
	//cublasScopy(h, n*m, thrust::raw_pointer_cast(v1.data()), m, thrust::raw_pointer_cast(result.data()), 1);
	//for (int i = 0; i < v1.size(); i++)
//		result[i] = v1[((i * m) % (n * m))];
	
	cublasSgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha, thrust::raw_pointer_cast(v1.data()), m, &beta, thrust::raw_pointer_cast(v1.data()), n, thrust::raw_pointer_cast(transpose.data()), n);
	cudaDeviceSynchronize(); 

	//cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, r, n, m, &alpha, thrust::raw_pointer_cast(v1.data()), m, thrust::raw_pointer_cast(transpose.data()), n, &beta, thrust::raw_pointer_cast(result.data()), n);
	//cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, r, n, m, &alpha, thrust::raw_pointer_cast(v1.data()), n, thrust::raw_pointer_cast(v2.data()), m, &beta, thrust::raw_pointer_cast(result.data()), r);
	/*
	std::cout << "Starting test" << std::endl << "=============================" << std::endl << std::endl;
	
	for (int a_rows = 2; a_rows <= 4; a_rows++)
	{
		for (int b_cols = 2; b_cols <= 4; b_cols++)
		{
			for (int a_cols = 2; a_cols <= 4; a_cols++)
			{

				for (int lda = 2; lda <= 4; lda++)
				{

					for (int ldb = 2; ldb <= 4; ldb++)
					{

						for (int ldc = 2; ldc <= 4; ldc++)
						{
						thrust::fill(result.begin(), result.end(), 0);
						cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, a_rows, b_cols, a_cols, &alpha, thrust::raw_pointer_cast(transpose.data()), lda, thrust::raw_pointer_cast(v2.data()), ldb, &beta, thrust::raw_pointer_cast(result.data()), ldc);
						cudaDeviceSynchronize(); 
						if (thrust::equal(result.begin(), result.end(), thrust::constant_iterator<float>(30)))
							std::cout << a_rows << b_cols << a_cols << lda << ldb << ldc << ": Success" << std::endl;
						}
					}
				}
			}
		}
	}
	*/
	cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, m, r, n, &alpha, thrust::raw_pointer_cast(transpose.data()), n, thrust::raw_pointer_cast(v2.data()), r, &beta, thrust::raw_pointer_cast(result.data()), m);
	cudaDeviceSynchronize();

	thrust::device_vector<float> new_result(m*r, 0);
	cublasSgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, r, m, &alpha, thrust::raw_pointer_cast(result.data()), m, &beta, thrust::raw_pointer_cast(result.data()), r, thrust::raw_pointer_cast(new_result.data()), r);
	cudaDeviceSynchronize(); 
	//std::cout << err << std::endl;
	
	cublasDestroy(h);
	
	printf("Size is %d\n", result.size());
	for (int i = 0; i < new_result.size(); i++)
	{
		std::cout << new_result[i] << " ";
	}

	return 0;
}
