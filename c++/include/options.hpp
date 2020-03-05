#ifndef OPTS_INCLUDED
#define OPTS_INCLUDED
class options 
{
	public:
		//Learning rate
		float alpha = 0.01;
		float beta = 0.90;
		float decay = 0.001;
		int epochs = 50;
		int n_x = 784;
		int n_h1 = 500;
		int n_h2 = 500;
		int n_o = 10;
		int batch_size = 500;
		int batches = 120;

};
#endif
