### Deep learning using the CPU and the GPU

This was a study of many things - the ultimate focus was neural networks and their performance using the gpu vs the cpu. This was also a refresher of C++ for me. Python implementation seems to return better results, but C++ is faster. Future research focus would be hyperparameter optimization.

Architecture is 784 inputs with 2 hidden layers of sizes 600 and 500, respectively. Output layer is 10 nodes. Alpha is 0.002 with a decay over every epoch. Momentum is used as well with a beta value of 0.9. Each network runs for 50 epochs for each batch size.

### Results
For reference:
* **P** is **P**ython
* **C** is **C**++

**
##### Tables
The table below showcases the timing differences between Python CPU and GPU versions, C++ CPU and GPU versions, and CPU and GPU differences.
![Image of time table](https://github.com/mrmattkennedy/mnist-neural-network/blob/master/results/figures/table_time.png)

The table below showcases the accuracy differences between Python CPU and GPU versions, C++ CPU and GPU versions, and CPU and GPU differences.
![Image of acc table](hhttps://github.com/mrmattkennedy/mnist-neural-network/blob/master/results/figures/table_acc.png)

##### Timing plots
The timings were what I was most interested in, so I plotted them using the matplotlib library.
Here's what the CPU and GPU timings look like compared visually at each batch size:
![Image of time plot](https://github.com/mrmattkennedy/neural-network-library/blob/master/python/data/figures/times.png)

And here's what the difference at each batch size looks like for batch sizes 1000 and under:
![Image of lte1000 plot](https://github.com/mrmattkennedy/neural-network-library/blob/master/python/data/figures/times_lte1000.png)


Additionally, in [this folder](/results/figures), there are plots of how each network does for each batch size.
