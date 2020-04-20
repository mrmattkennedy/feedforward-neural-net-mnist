def get_results():
    import mnist_nn
    import mnist_nn_gpu
    
    mnist_nn.save_results()
    mnist_nn_gpu.save_results()

def visualize():
    import numpy as np
    import matplotlib.pyplot as plt
    import pdb
    """
    Figures created here:
    Table for each batch size with various statistics
    Figure for each batch size comparing accuracy at each epoch for CPU vs GPU
    Figure for CPU and GPU, as well as charting diff
    """
    
    #Load in data
    cpu_times, gpu_times, cpu_accuracy, gpu_accuracy = load_data()
    batch_sizes = list(cpu_times.keys())
    
    #Get time data
    cpu_times = list(cpu_times.values())
    gpu_times = list(gpu_times.values())
    diff_times = [a - b for a, b in zip(cpu_times, gpu_times)]

    #Get accuracy data
    cpu_accuracy = list(cpu_accuracy.values())
    cpu_max = [[max(item), item.index(max(item))] for item in cpu_accuracy]
    gpu_accuracy = list(gpu_accuracy.values())
    gpu_max = [[max(item), item.index(max(item))] for item in gpu_accuracy]

    #Set data in np array for table
    data = np.array([cpu_times,
                     gpu_times,
                     diff_times,
                     [item[0] for item in cpu_max],
                     [item[1] for item in cpu_max],
                     [item[0] for item in gpu_max],
                     [item[1] for item in gpu_max]]).T

    #Get data in text format
    n_rows = data.shape[0]
    cell_text = []
    for row in range(n_rows):
        cell_text.append(['%1.3f' % x for x in data[row]])
    
    #Get rows and cols for table
    columns = ('CPU Time (s)', 'GPU Time (s)', 'Diff (s)', 'CPU Acc Max (%)', 'Max index', 'GPU Acc Max (%)', 'Max Index')
    row_colors = plt.cm.BuPu(np.linspace(0, 0.5, n_rows))
    col_colors = np.array([192/255,192/255,192/255, 1])
    col_colors = np.repeat(col_colors.reshape((1, col_colors.shape[0])), len(columns), axis=0)

    #Create table
    plt.figure(figsize=(10.8,9.4)).canvas.set_window_title('CPU vs GPU MNIST Neural Network')
    plt.table(cellText=cell_text,
              rowLabels=batch_sizes,
              rowColours=row_colors,
              colLabels=columns,
              colColours=col_colors,
              loc='center')
    ax = plt.gca()
    ax.axis('off')
    plt.savefig('data\\figures\\table.png')

    #Create plots of time for CPU vs GPU
    plt.clf()
    plt.figure(figsize=(10.8,9.4)).canvas.set_window_title('CPU vs GPU MNIST Neural Network Times')
    l1, = plt.plot(batch_sizes, cpu_times, '-o')
    l2, = plt.plot(batch_sizes, gpu_times, '-s')
    plt.legend((l1, l2), ('CPU Times', 'GPU Times'))
    plt.xlabel('Batch size')
    plt.ylabel('Time (s)')    
    plt.grid()
    plt.savefig('data\\figures\\times.png')

    #Create plot of time difference
    plt.clf()
    plt.figure(figsize=(10.8,9.4)).canvas.set_window_title('CPU vs GPU MNIST Neural Network Times')
    plt.plot(batch_sizes, diff_times, '-o')
    plt.xlabel('Batch size')
    plt.ylabel('Time difference (s)')
    plt.grid()
    plt.savefig('data\\figures\\times_diff.png')

    #Create plot for each batch size comparing CPU and GPU accuracy
    x_axis = list(range(1, len(cpu_accuracy[0]) + 1))
    min_y = max(min([min(item) for item in cpu_accuracy]) - 0.05, 0.0)
    for item in range(len(cpu_accuracy)):
        plt.clf()
        plt.close()
        plt.figure(figsize=(10.8,9.4)).canvas.set_window_title('CPU vs GPU MNIST Neural Network Accuracy, Batch Size {}'.format(batch_sizes[item]))
        l1, = plt.plot(x_axis, cpu_accuracy[item], '-o')
        l2, = plt.plot(x_axis, gpu_accuracy[item], '-s')
        plt.legend((l1, l2), ('CPU Accuracy', 'GPU Accuracy'))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.ylim((min_y, 1.0))
        plt.grid()
        plt.savefig('data\\figures\\size_{}_acc.png'.format(batch_sizes[item]))
    
def load_data():
    import collections
        
    #Define paths
    cpu_times_path = 'data\\cpu_times.dat'
    cpu_accuracy_path = 'data\\cpu_accuracy.dat'
    gpu_times_path = 'data\\gpu_times.dat'
    gpu_accuracy_path = 'data\\gpu_accuracy.dat'

    #Load CPU times
    cpu_times = {}    
    with open(cpu_times_path, 'r') as file:
        for line in file:
            key, value = line.split(',')
            cpu_times[int(key)] = float(value)
    cpu_times = collections.OrderedDict(sorted(cpu_times.items()))

    #Load GPU times
    gpu_times = {}
    with open(gpu_times_path, 'r') as file:
        for line in file:
            key, value = line.split(',')
            gpu_times[int(key)] = float(value)
    gpu_times = collections.OrderedDict(sorted(gpu_times.items()))

    #Load CPU accuracies
    cpu_accuracy = collections.defaultdict(list)
    with open(cpu_accuracy_path, 'r') as file:
        for line in file:
            if line.strip():
                key, value = line.split(',')
                cpu_accuracy[int(key)].append(float(value))
    cpu_accuracy = collections.OrderedDict(sorted(cpu_accuracy.items()))

    #Load GPU accuracies
    gpu_accuracy = collections.defaultdict(list)
    with open(gpu_accuracy_path, 'r') as file:
        for line in file:
            if line.strip():
                key, value = line.split(',')
                gpu_accuracy[int(key)].append(float(value))
    gpu_accuracy = collections.OrderedDict(sorted(gpu_accuracy.items()))

    return cpu_times, gpu_times, cpu_accuracy, gpu_accuracy

    
visualize()
