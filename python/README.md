Python implementation of a neural network. Utilizes decay for the learning rate, momentum with gradient descent, and shuffling data sets with batches. Reporting over 99.9% accuracy after 150 epochs.

Created GPU version as well to compare time and accuracy with CUDA library. See [this folder](../results/figures) for figures comparing GPU and CPU timings, accuracies, and more. I'll list some interesting figures below, but theres quite a few more to see in that folder. Points of interest:
* The timings at each batch size
* The difference in each accuracy (visualization provided for each batch size)
* GPU accuracy mostly peaks above 0.99 accuracy at each size, whereas CPU never manages to do this
* Gap in CPU vs GPU accuracy closes as batch size increases

