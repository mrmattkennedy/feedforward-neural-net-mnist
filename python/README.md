Python implementation of a neural network. Utilizes decay for the learning rate, momentum with gradient descent, and shuffling data sets with batches. Reporting over 99.9% accuracy currently after 150 epochs.

Created GPU version as well to compare time and accuracy with CUDA library. See [this folder](../master/python/data/figures) for figures comparing GPU and CPU timings, accuracies, and more. I'll list some interesting figures below, but theres quite a few more to see in that folder.

The table below showcases the timing difference, as well as accuracy difference for each batch size specified.
![Image of table](https://github.com/mrmattkennedy/neural-network-library/blob/master/python/data/figures/table.png)

The timings were what I was most interested in, so I plotted them using the matplotlib library.
Here's what the CPU and GPU timings look like compared visually:
![Image of timings](https://github.com/mrmattkennedy/neural-network-library/blob/master/python/data/figures/times.png)

And here's what the difference at each batch size looks like:
![Image of diff](https://github.com/mrmattkennedy/neural-network-library/blob/master/python/data/figures/times_diff.png)
