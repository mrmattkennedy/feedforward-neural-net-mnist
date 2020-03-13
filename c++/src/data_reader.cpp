#include <assert.h>
#include <string>
#include <fstream>
#include <iterator>
#include <vector>
#include "data_reader.hpp"

data_reader::data_reader
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
#
