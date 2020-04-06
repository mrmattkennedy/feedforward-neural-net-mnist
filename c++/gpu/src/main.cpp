#include "neural_net.hpp"


int main(int argc, char** argv)
{
	std::string base_path = "..\\..\\MNIST data\\";
	neural_net nn(base_path);
	nn.train();


	/*
	int n = 2, m = 2, r = 2;
	thrust::device_vector<float> v1(n*m, 1);
	thrust::device_vector<float> v2(m*r, 1);
	v1[0] = 1;
	v1[1] = 2;
	v1[2] = 3;
	v1[3] = 4;
	v2[0] = 1;
	v2[1] = 2;
	v2[2] = 3;
	v2[3] = 4;
	thrust::device_vector<float> result(n*r, 0);
	cublasHandle_t h;
	cublasCreate(&h);
	float alpha = 1.0f, beta=0.0f;

	cudaDeviceSynchronize();
	cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, r, m, &alpha, thrust::raw_pointer_cast(v1.data()), n, thrust::raw_pointer_cast(v2.data()), m, &beta, thrust::raw_pointer_cast(result.data()), n);
	cudaDeviceSynchronize(); 
	cublasDestroy(h);
	for (int i = 0; i < result.size(); i++)
	{
		std::cout << result[i] << " ";
		if (i == 1)
			std::cout << std::endl;
	}
*/
	return 0;
}
