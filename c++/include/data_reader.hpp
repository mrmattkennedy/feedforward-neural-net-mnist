#ifndef DATA_INCLUDED
#define DATA_INCLUDED

#include <assert.h>
#include <string>
#include <fstream>
#include <iterator>
#include <vector>

class Data_Reader
{
	private:
		std::string base_path = "..\\MNIST data\\";
		std::string train_data_path = base_path + "train-images.idx3-ubyte";
		std::string train_labels_path = base_path + "train-labels.idx1-ubyte";
		std::string test_data_path = base_path + "t10k-images.idx3-ubyte";
		std::string test_labels_path = base_path + "t10k-images.idx1-ubyte";

  		std::vector<std::vector<unsigned char>> m_images;
		std::vector<int> m_labels;

		unsigned int m_size;
		unsigned short int rows;
		unsigned short int cols;

	public:
		int to_int(char* p)
		{
		  return ((p[0] & 0xff) << 24) | ((p[1] & 0xff) << 16) |
			 ((p[2] & 0xff) <<  8) | ((p[3] & 0xff) <<  0);
		}


		int get_train_images()
		{
			
			std::ifstream ifs(train_data_path, std::ios::in | std::ios::binary);

			//idx file format chunked into 4 bytes each.
			//Magic number first - 3rd byte is data type, 4th is # dimensions.
			char p[4];

			ifs.read(p, 4);
			assert((p[0] == 0) && (p[1] == 1) && (p[2] == 8) && (p[3] == 3));
			int dims = p[3];
			
			//Get sizes
			ifs.read(p, 4);
			m_size = to_int(p);
			ifs.read(p, 4);
			rows = to_int(p);
			ifs.read(p, 4);
			cols = to_int(p);
			
			//Read elements in
			char q[rows*cols];
			for (int i = 0; i < m_size; i++)
			{
				//Read in rows*cols bytes, assign to a new vector image, push back image on images
				ifs.read(q, rows*cols);
				std::vector<unsigned char> image;
				image.reserve(rows*cols);
				std::copy(q, q+(rows*cols), std::back_inserter(image));	
				m_images.push_back(image);
			}
//			
			ifs.close();
		}
};
#endif