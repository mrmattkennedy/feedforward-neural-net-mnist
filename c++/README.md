C++ implementation of a neural network. Utilizes decay for the learning rate, momentum with gradient descent, and shuffling data sets with batches.

Created GPU version as well to compare time and accuracy with CUDA library. See [this folder](../results/figures) for figures comparing GPU and CPU timings, accuracies, and more. I'll list some interesting figures below, but theres quite a few more to see in that folder. Points of interest:
* The timings at each batch size
* The difference in each accuracy (visualization provided for each batch size)
* GPU accuracy does fairly poorly compared to python equivalent
* CPU is faster than GPU to start, but GPU takes over quickly. Also faster than python counterpart towards the larger batch sizes.

