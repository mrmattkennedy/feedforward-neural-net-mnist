#ifndef DATA_INCLUDED
#define DATA_INCLUDED

#include <string>
#include <vector>

class data_reader
{
	private:
		std::string base_path;
		std::string train_data_path;
		std::string train_labels_path;
		std::string test_data_path;
		std::string test_labels_path;

  		std::vector<std::vector<unsigned char>> m_images;
		std::vector<int> m_labels;

		unsigned int m_size;
		unsigned short int n_rows;
		unsigned short int n_cols;
		
		void load_images();
		void load_labels();
		int to_int(char* p);

	public:
		data_reader(std::string base_path);
		~data_reader();

		int size() { return m_size; }
		int rows() { return n_rows; }
		int cols() { return n_cols; }

		std::vector<unsigned char> images(int id) { return m_images[id]; }
		int labels(int id) { return m_labels[id]; }
};
#endif
