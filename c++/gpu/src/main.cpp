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



//thrust::device_vector<float> matrix_multiply(thrust::device_vector<float> A, thrust::device_vector<float> B, int a_rows, int a_cols, int b_rows, int b_cols, int op);

int main(int argc, char** argv)
{
	std::string base_path = "..\\..\\MNIST data\\";
	neural_net nn(base_path);
	nn.train();
/*	
//	int n = 70000;
//	thrust::device_vector<float> v1(70000, 1);
	thrust::device_vector<float> v2(m*r, 1);
	
	v2[0] = 7;
	v2[1] = 9;
	v2[2] = 11;
	v2[3] = 13;
	v2[4] = 8;
	v2[5] = 10;
	v2[6] = 12;
	v2[7] = 14;
	v1[0] = 1;
	v1[1] = 2;
	v1[2] = 3;
	v1[3] = 4;
	v1[4] = 5;
	v1[5] = 6;

	cublasHandle_t h;
	cublasCreate(&h);
	thrust::device_vector<float> result(n*r, 0);
	float alpha = 1.0f, beta=0.0f;
	
	cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, n, r, m, &alpha, thrust::raw_pointer_cast(v1.data()), m, thrust::raw_pointer_cast(v2.data()), r, &beta, thrust::raw_pointer_cast(result.data()), n);
	cudaDeviceSynchronize();
	for (int i = 0; i < n*r; i++)
		std::cout << result[i] << " ";
	std::cout << std::endl;
	*/
	/*
	v2[0] = 13;
	v2[1] = 14;
	v2[2] = 15;
	v2[3] = 16;
	v2[4] = 17;
	v2[5] = 18;
	v2[6] = 19;
	v2[7] = 20;
	*/
	//Transpose
	/*
	v1[0] = 1;
	v1[1] = 4;
	v1[2] = 7;
	v1[3] = 10;
	v1[4] = 2;
	v1[5] = 5;
	v1[6] = 8;
	v1[7] = 11;
	v1[8] = 3;
	v1[9] = 6;
	v1[10] = 9;
	v1[11] = 12;
	*/
	
	//Regular
	/*
	v1[0] = 1;
	v1[1] = 2;
	v1[2] = 3;
	v1[3] = 4;
	v1[4] = 5;
	v1[5] = 6;
	v1[6] = 7;
	v1[7] = 8;
	v1[8] = 9;
	v1[9] = 10;
	v1[10] = 11;
	v1[11] = 12;
	*/
	/*
	thrust::device_vector<float> transpose(4*2, 0);
	//thrust::device_vector<float> result(3*4, 0);
	cublasHandle_t h;
	cublasCreate(&h);
	float alpha = 1.0f, beta=0.0f;
	
	thrust::fill(v1.begin(), v1.end(), 1);
	thrust::fill(v2.begin(), v2.end(), 1);
	//auto result = matrix_multiply(v1, v2, 70000, 500, 70000, 10, 0x02);
	for (int i = 0; i < 10; i++)
		std::cout << result[i] << " ";
	std::cout << std::endl;
	*/

	/*
	//A x B, result is mxr
	//Multiply
	cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, n, r, m, &alpha, thrust::raw_pointer_cast(v1.data()), m, thrust::raw_pointer_cast(v2.data()), r, &beta, thrust::raw_pointer_cast(result.data()), n);
	cudaDeviceSynchronize();
	//Transpose result to row major
	thrust::device_vector<float> new_result(n*r, 0);
	
	cublasSgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, r, n, &alpha, thrust::raw_pointer_cast(result.data()), n, &beta, thrust::raw_pointer_cast(result.data()), r, thrust::raw_pointer_cast(new_result.data()), r);
	cudaDeviceSynchronize(); 
	*/
	/*
	//A.T x B, result is mxr, transpose is nxm
	//Transpose
	cublasSgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha, thrust::raw_pointer_cast(v1.data()), m, &beta, thrust::raw_pointer_cast(v1.data()), n, thrust::raw_pointer_cast(transpose.data()), n);
	cudaDeviceSynchronize(); 
	//Multiply
	cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, m, r, n, &alpha, thrust::raw_pointer_cast(transpose.data()), n, thrust::raw_pointer_cast(v2.data()), r, &beta, thrust::raw_pointer_cast(result.data()), m);
	cudaDeviceSynchronize();
	//Transpose result to row major
	thrust::device_vector<float> new_result(m*r, 0);
	cublasSgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, r, m, &alpha, thrust::raw_pointer_cast(result.data()), m, &beta, thrust::raw_pointer_cast(result.data()), r, thrust::raw_pointer_cast(new_result.data()), r);
	cudaDeviceSynchronize(); 
	for (int i = 0; i < new_result.size(); i++)
		std::cout << new_result[i] << " ";
	std::cout << std::endl;
	*/
	
	
	//A x B.T, result is nxr, transpose is mxr
	//Transpose
	/*
	cublasSgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, r, m, &alpha, thrust::raw_pointer_cast(v2.data()), m, &beta, thrust::raw_pointer_cast(v2.data()), r, thrust::raw_pointer_cast(transpose.data()), r);
	cudaDeviceSynchronize(); 
	//for (int i = 0; i < transpose.size(); i++)
	//	std::cout << transpose[i] << " ";
	//std::cout << std::endl;
	
	
	//Multiply
	cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, n, r, m, &alpha, thrust::raw_pointer_cast(v1.data()), m, thrust::raw_pointer_cast(transpose.data()), r, &beta, thrust::raw_pointer_cast(result.data()), n);
	cudaDeviceSynchronize();
	//Transpose result to row major
	thrust::device_vector<float> new_result(n*r, 0);	
	cublasSgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, r, n, &alpha, thrust::raw_pointer_cast(result.data()), n, &beta, thrust::raw_pointer_cast(result.data()), r, thrust::raw_pointer_cast(new_result.data()), r);
	cudaDeviceSynchronize(); 
	*/
	
	/*
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
						cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, a_rows, b_cols, a_cols, &alpha, thrust::raw_pointer_cast(v1.data()), lda, thrust::raw_pointer_cast(transpose.data()), ldb, &beta, thrust::raw_pointer_cast(result.data()), ldc);
						cudaDeviceSynchronize(); 
						if (result[0] == 23)
						{
							std::cout << a_rows << b_cols << a_cols << lda << ldb << ldc << " : ";
							for (int i = 0; i < result.size(); i++)
								std::cout << result[i] << " ";
							std::cout << std::endl;
						}
						}
					}
				}
			}
		}
	}
	
	cublasDestroy(h);
	*/
	return 0;
}

/*
thrust::device_vector<float> matrix_multiply(thrust::device_vector<float> A, thrust::device_vector<float> B, int a_rows, int a_cols, int b_rows, int b_cols, int op)
{
	int n = a_rows, m = a_cols, r = b_cols;
	float alpha = 1.0f, beta=0.0f;
	thrust::device_vector<float> new_result;

	cublasHandle_t h;
	cublasCreate(&h);

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
*/
