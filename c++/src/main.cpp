#include <iostream>
#include "options.hpp"
#include "mnist_loader.hpp"
#include "eigen-3.3.7/Eigen/Dense"

int main(int argc, char** argv)
{
	options opts;
	std::string data_path = "..\\MNIST data\\";
	mnist_loader train(data_path + "train-images.idx3-ubyte",
		"dataset/train-labels.idx1-ubyte");
	mnist_loader test("dataset/t10k-images.idx3-ubyte",
		"dataset/t10k-labels.idx1-ubyte");
//	mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset = 
//		mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(data_path);
	

//	Eigen::MatrixXd m = Eigen::MatrixXd::Random(3,3);
//	Eigen::Matrix<unsigned char, 60000, 28, 28> test = Eigen::Map<Eigen::Matrix<unsigned char, 60000, 28, 28>, Eigen::Unaligned, Eigen::Stride<28, 28> >(dataset.training_images);
//	auto temp = std::vector<Eigen::Vector4f,Eigen::aligned_allocator<Eigen::Vector4f> >
//	std::cout << "m = " << std::endl << m << std::endl;
//	std::cout << "Nbr of training images = " << dataset.training_images[0] << std::endl;
//	std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
//	std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
//	std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
//	create_matrix(dataset);
	return 1;
}

/*
void create_matrix(mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset)
{
	//28x28 matrix
	for (auto& vec : dataset.training_images) {
		Eigen::Matrix<unsigned char, 28, 28> temp;
		int row = 0;
		int col = 0;
		for (auto& v : vec)
		{
			temp[row][col] = v;
			if ((col + 1) % 28 == 0)
				row++;

			col = (col + 1) % 28;
		}
		break;
 	}
}
*/
