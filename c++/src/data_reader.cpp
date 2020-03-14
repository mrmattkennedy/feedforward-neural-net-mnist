#include <assert.h>
#include <string>
#include <fstream>
#include <iterator>
#include <vector>
#include "data_reader.hpp"

data_reader::data_reader(std::string base_path)
{

		this->base_path = base_path;
		train_data_path = base_path + "train-images.idx3-ubyte";
		train_labels_path = base_path + "train-labels.idx1-ubyte";
		test_data_path = base_path + "t10k-images.idx3-ubyte";
		test_labels_path = base_path + "t10k-images.idx1-ubyte";
}

data_reader::~data_reader()
{

}



int data_reader::to_int(char* p)
{
  return ((p[0] & 0xff) << 24) | ((p[1] & 0xff) << 16) |
	 ((p[2] & 0xff) <<  8) | ((p[3] & 0xff) <<  0);
}


void data_reader::load_images()
{
	
	std::ifstream ifs(train_data_path, std::ios::in | std::ios::binary);

	//idx file format chunked into 4 bytes each.
	//Magic number first - 3rd byte is data type, 4th is # dimensions.
	char p[4];

	ifs.read(p, 4);
	int magic_number = to_int(p);
	assert(magic_number == 0x803);

	//Get sizes
	ifs.read(p, 4);
	m_size = to_int(p);
	ifs.read(p, 4);
	n_rows = to_int(p);
	ifs.read(p, 4);
	n_cols = to_int(p);
	
	//Read elements in
	char q[n_rows*n_cols];
	for (int i = 0; i < m_size; i++)
	{
		//Read in rows*cols bytes, assign to a new vector image, push back image on images
		ifs.read(q, n_rows*n_cols);
		std::vector<unsigned char> image;
		image.reserve(n_rows*n_cols);
		std::copy(q, q+(n_rows*n_cols), std::back_inserter(image));	
		m_images.push_back(image);
	}
	ifs.close();
}

