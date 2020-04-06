#include <assert.h>
#include <string>
#include <fstream>
#include <iterator>
#include <vector>
#include <algorithm>
#include "data_reader.hpp"

data_reader::data_reader(std::string base_path)
{

		this->base_path = base_path;
		train_data_path = base_path + "train-images.idx3-ubyte";
		train_labels_path = base_path + "train-labels.idx1-ubyte";
		test_data_path = base_path + "t10k-images.idx3-ubyte";
		test_labels_path = base_path + "t10k-labels.idx1-ubyte";

		load_images(train_data_path);
		load_images(test_data_path);
		load_labels(train_labels_path);
		load_labels(test_labels_path);
}

data_reader::~data_reader()
{

}



int data_reader::to_int(char* p)
{
  return ((p[0] & 0xff) << 24) | ((p[1] & 0xff) << 16) |
	 ((p[2] & 0xff) <<  8) | ((p[3] & 0xff) <<  0);
}


void data_reader::load_images(std::string data_path)
{
	
	std::ifstream ifs(data_path, std::ios::in | std::ios::binary);

	//idx file format chunked into 4 bytes each.
	//Magic number first - 3rd byte is data type, 4th is # dimensions.
	char p[4];

	ifs.read(p, 4);
	int magic_number = to_int(p);
	assert(magic_number == 0x803);

	//Get sizes
	ifs.read(p, 4);
	int size = to_int(p);
	ifs.read(p, 4);
	n_rows = to_int(p);
	ifs.read(p, 4);
	n_cols = to_int(p);
	

	//Read elements in. Compilers other than g++ don't allow arrays of variable size.
	char q[28*28];
	for (int i = 0; i < size; i++)
	{
		//Read in rows*cols bytes, assign to a new vector image, push back image on images
		ifs.read(q, n_rows*n_cols);
		std::vector<unsigned char> image;
		std::copy(q, q+(n_rows*n_cols), std::back_inserter(image));
		//Reading directly in to a double will use more than 1 byte, giving wrong inputs
		std::vector<float> double_image(image.begin(), image.end());
		//Use a 1D array for thrust device vector
		m_images.insert(m_images.end(), double_image.begin(), double_image.end());
//		m_images.push_back(double_image);
	}
	
	ifs.close();
}

void data_reader::load_labels(std::string labels_path)
{
	
	std::ifstream ifs(labels_path, std::ios::in | std::ios::binary);

	//idx file format chunked into 4 bytes each.
	//Magic number first - 3rd byte is data type, 4th is # dimensions.
	char p[4];

	ifs.read(p, 4);
	int magic_number = to_int(p);
	assert(magic_number == 0x801);

	ifs.read(p, 4);
	int size = to_int(p);
	for (int i = 0; i < size; i++)
	{
		ifs.read(p, 1);
		int label = p[0];
		m_labels.push_back(label);
	}
}
