#ifndef DATA_INCLUDED
#define DATA_INCLUDED

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

	public:
		int to_int(char* p)
		{
		  return ((p[0] & 0xff) << 24) | ((p[1] & 0xff) << 16) |
			 ((p[2] & 0xff) <<  8) | ((p[3] & 0xff) <<  0);
		}

		int get_train_images()
		{
			
			std::ifstream ifs(train_data_path, std::ios::in | std::ios::binary);
			char p[4];

			ifs.read(p, 4);
			int magic_number = to_int(p);
			std::cout << "num is " << magic_number << std::endl;
			
			ifs.close();
		}
};
#endif
