This is the C++ implementation of a cnn for the MNIST data set.  Can just use make, but better is:
g++ -I include/ -mavx -g -O3 -DNDEBUG -fopenmp src/*.cpp
