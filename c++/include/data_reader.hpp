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

		unsigned short int n_rows;
		unsigned short int n_cols;

		
		void load_images(std::string data_path);
		void load_labels(std::string labels_path);
		int to_int(char* p);

	public:
		data_reader(std::string base_path);
		~data_reader();

		int size() { return m_images.size(); }
		int rows() { return n_rows; }
		int cols() { return n_cols; }

  		std::vector<std::vector<double>> m_images;
		std::vector<int> m_labels;
};
#endif
