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
		int get_train_images()
		{
			/*
			// open the file:
			std::ifstream file(train_data_path, std::ios::binary);
			
			std::cout << "good: " << file.good() << std::endl;
			// Stop eating new lines in binary mode!!!
			file.unsetf(std::ios::skipws);

			// get its size:
			std::streampos fileSize;

			file.seekg(0, std::ios::end);
			fileSize = file.tellg();
			file.seekg(0, std::ios::beg);

			// reserve capacity
			std::vector<char> vec;
			std::cout << "Filesize: " << fileSize << std::endl;
			vec.reserve(fileSize);

			// read the data:
			vec.insert(vec.begin(),
					std::istream_iterator<char>(file),
					std::istream_iterator<char>());
			*/

			std::ifstream input(train_data_path, std::ios::binary);
			std::vector<char> bytes(
				(std::istreambuf_iterator<char>(input)),
				(std::istreambuf_iterator<char>()));

			input.close();
			std::cout << "Size: " << bytes.size() << std::endl;
		}
};
#endif
