import pdb
import numpy as np
import matplotlib.pyplot as plt

def get_results():
    """
    Calls functions in each NN file to get results.
    Starts with python, calls run file for c++ versions
    """
    #Get python results
    import mnist_nn
    import mnist_nn_gpu
    mnist_nn.save_results()
    mnist_nn_gpu.save_results()

    #Get cpp results
    import subprocess
    subprocess.call(['c++//./run.sh'])

def visualize():
    """
    Figures created here:
    Table comparing Python/C++, CPU/GPU times and another for accuracy
    Figure for each batch size comparing accuracy at each epoch for Python/C++, CPU/GPU
    Figure for time difference for Python/C++, CPU/GPU
    """
    
    #Load in data
    times, accuracies = load_data()
    batch_sizes = list(times[0].keys())

    #Create tables
    create_tables(times, accuracies, batch_sizes)

    #Create time plots
    create_time_plots(times, batch_sizes)

    #Create plot of accuracy at each batch size
    create_batch_plots(accuracies, batch_sizes)

    

def create_tables(times, accuracies, batch_sizes):
    """
    Create tables
    Start with Times table, load in the data and put into useful lists.
    Load lists into numpy array, then create cell text.
    Create columns, column colors, then create the table and save it.
    Repeat for accuracy table.
    """
    #Get time data
    p_cpu_times = list(times[0].values())
    p_gpu_times = list(times[1].values())
    c_cpu_times = list(times[2].values())
    c_gpu_times = list(times[3].values())

    #Get differences in times
    p_diff_times = [a - b for a, b in zip(p_cpu_times, p_gpu_times)]
    c_diff_times = [a - b for a, b in zip(c_cpu_times, c_gpu_times)]
    cpu_diff_times = [a - b for a, b in zip(p_cpu_times, c_cpu_times)]
    gpu_diff_times = [a - b for a, b in zip(p_gpu_times, c_gpu_times)]

    #Set data in np array for table
    data = np.array([p_cpu_times,
                     p_gpu_times,
                     p_diff_times,
                     c_cpu_times,
                     c_gpu_times,
                     c_diff_times,
                     cpu_diff_times,
                     gpu_diff_times]).T

    #Get data in text format
    n_rows = data.shape[0]
    cell_text = []
    for row in range(n_rows):
        cell_text.append(['%1.3f' % x for x in data[row]])
    
    #Get rows and cols for table
    columns = ('P CPU Time (s)', 'P GPU Time (s)', 'P Diff (s)', 'C CPU Time (s)', 'C GPU Time (s)', 'C Diff (s)', 'CPU Diff (s)', 'GPU Diff (s)')
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
    plt.savefig('results\\figures\\table_time.png')


    #Get accuracy table
    #Get accuracy data
    p_cpu_accuracy = list(accuracies[0].values())
    p_gpu_accuracy = list(accuracies[1].values())
    c_cpu_accuracy = list(accuracies[2].values())
    c_gpu_accuracy = list(accuracies[3].values())

    #Get max of each batch
    p_cpu_max = [max(x) for x in p_cpu_accuracy]
    p_gpu_max = [max(x) for x in p_gpu_accuracy]
    c_cpu_max = [max(x) for x in c_cpu_accuracy]
    c_gpu_max = [max(x) for x in c_gpu_accuracy]

    #Get differences in accuracies
    p_diff_acc = [a - b for a, b in zip(p_cpu_max, p_gpu_max)]
    c_diff_acc = [a - b for a, b in zip(c_cpu_max, c_gpu_max)]
    cpu_diff_acc = [a - b for a, b in zip(p_cpu_max, c_cpu_max)]
    gpu_diff_acc = [a - b for a, b in zip(p_gpu_max, c_gpu_max)]

    #Set data in np array for table
    data = np.array([p_cpu_max,
                     p_gpu_max,
                     p_diff_acc,
                     c_cpu_max,
                     c_gpu_max,
                     c_diff_acc,
                     cpu_diff_acc,
                     gpu_diff_acc]).T

    #Get data in text format
    n_rows = data.shape[0]
    cell_text = []
    for row in range(n_rows):
        cell_text.append(['%1.3f' % x for x in data[row]])
    
    #Get rows and cols for table
    columns = ('P CPU Acc (%)', 'P GPU Acc (%)', 'P Diff (%)', 'C CPU Acc (%)', 'C GPU Time (%)', 'C Diff (%)', 'CPU Diff (%)', 'GPU Diff (%)')

    #Create table
    plt.clf()
    plt.figure(figsize=(10.8,9.4)).canvas.set_window_title('CPU vs GPU MNIST Neural Network')
    plt.table(cellText=cell_text,
              rowLabels=batch_sizes,
              rowColours=row_colors,
              colLabels=columns,
              colColours=col_colors,
              loc='center')
    ax = plt.gca()
    ax.axis('off')
    plt.savefig('results\\figures\\table_acc.png')



def create_time_plots(times, batch_sizes):
    """
    Create time plots
    Load in timing information into useful lists, then create 3 plots:
    Comparison of timings all 4 data sets on a plot.
    Comparison of timings differences all 4 data sets on a plot for batch sizes less than or equal to 1000.
       -Diff is CPU time - GPU time.
    Comparison of timings all 4 data sets on a plot for batch sizes less than or equal to 1000.
    """
    #Get time data
    p_cpu_times = list(times[0].values())
    p_gpu_times = list(times[1].values())
    c_cpu_times = list(times[2].values())
    c_gpu_times = list(times[3].values())

    #Get differences in times
    p_diff_times = [a - b for a, b in zip(p_cpu_times, p_gpu_times)]
    c_diff_times = [a - b for a, b in zip(c_cpu_times, c_gpu_times)]
    cpu_diff_times = [a - b for a, b in zip(p_cpu_times, c_cpu_times)]
    gpu_diff_times = [a - b for a, b in zip(p_gpu_times, c_gpu_times)]
    
    #Create plots of time for CPU vs GPU
    plt.clf()
    plt.figure(figsize=(10.8,9.4)).canvas.set_window_title('CPU vs GPU MNIST Neural Network Times, Python and C++')
    l1, = plt.plot(batch_sizes, p_cpu_times, '-o')
    l2, = plt.plot(batch_sizes, p_gpu_times, '-s')
    l3, = plt.plot(batch_sizes, c_cpu_times, '-o')
    l4, = plt.plot(batch_sizes, c_gpu_times, '-s')
    plt.legend((l1, l2, l3, l4), ('P CPU Times', 'P GPU Times', 'C CPU Times', 'C GPU Times'))
    plt.xlabel('Batch size')
    plt.ylabel('Time (s)')    
    plt.grid()
    plt.savefig('results\\figures\\times.png')

    #Create sub 1000 batch sizes time plot
    plt.clf()
    plt.figure(figsize=(10.8,9.4)).canvas.set_window_title('CPU vs GPU MNIST Neural Network Times, batch size <= 1000, Python and C++')
    temp_batch_size = [x for x in batch_sizes if x <= 1000]
    l1, = plt.plot(temp_batch_size, p_cpu_times[:len(temp_batch_size)], '-o')
    l2, = plt.plot(temp_batch_size, p_gpu_times[:len(temp_batch_size)], '-s')
    l3, = plt.plot(temp_batch_size, c_cpu_times[:len(temp_batch_size)], '-o')
    l4, = plt.plot(temp_batch_size, c_gpu_times[:len(temp_batch_size)], '-s')
    plt.legend((l1, l2, l3, l4), ('P CPU Times', 'P GPU Times', 'C CPU Times', 'C GPU Times'))
    plt.xlabel('Batch size')
    plt.ylabel('Time (s)')    
    plt.grid()
    plt.savefig('results\\figures\\times_lte1000.png')


    #Create plot of time difference
    plt.clf()
    plt.figure(figsize=(10.8,9.4)).canvas.set_window_title('CPU vs GPU MNIST Neural Network Times Difference, Python and C++')
    l1, = plt.plot(batch_sizes, p_diff_times, '-o')
    l2, = plt.plot(batch_sizes, c_diff_times, '-s')
    plt.legend((l1, l2), ('P Diff', 'C Diff'))
    plt.xlabel('Batch size')
    plt.ylabel('Time difference (s)')
    plt.grid()
    plt.savefig('results\\figures\\times_diff.png')


def create_batch_plots(accuracies, batch_sizes):
    """
    Create accuracy plots for each batch size
    Load in accuracy information into useful lists, then create a plot.
    Each plot has the accuracy for all 4 datasets listed
    """
    #Get accuracy data
    p_cpu_accuracy = list(accuracies[0].values())
    p_gpu_accuracy = list(accuracies[1].values())
    c_cpu_accuracy = list(accuracies[2].values())
    c_gpu_accuracy = list(accuracies[3].values())
    
    #Create plot for each batch size comparing CPU and GPU accuracy
    x_axis = list(range(1, len(p_cpu_accuracy[0]) + 1))
    #Get the min of each key (batch size) for all 4 datasets, then get the minimum of those 4, then get the minimum again.
    min_y = max(min([min(x) for x in [[min(dataset[key]) for key in dataset] for dataset in accuracies]]) - 0.05, 0.0)
    for item in range(len(p_cpu_accuracy)):
        plt.clf()
        plt.close()
        plt.figure(figsize=(10.8,9.4)).canvas.set_window_title('CPU vs GPU MNIST Neural Network Accuracy, Python and C++, Batch Size {}'.format(batch_sizes[item]))
        l1, = plt.plot(x_axis, p_cpu_accuracy[item], '-o')
        l2, = plt.plot(x_axis, p_gpu_accuracy[item], '-s')
        l3, = plt.plot(x_axis, c_cpu_accuracy[item], '-o')
        l4, = plt.plot(x_axis, c_gpu_accuracy[item], '-s')
        plt.legend((l1, l2, l3, l4), ('P CPU Acc', 'P GPU Acc', 'C CPU Acc', 'C GPU Acc'))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.ylim((min_y, 100.0))
        plt.grid()
        plt.savefig('results\\figures\\size_{}_acc.png'.format(batch_sizes[item]))
        
def load_data():
    import collections
        
    #Define paths
    p_cpu_times_path = 'results\\python\\cpu_times.dat'
    p_cpu_accuracy_path = 'results\\python\\cpu_accuracy.dat'
    p_gpu_times_path = 'results\\python\\gpu_times.dat'
    p_gpu_accuracy_path = 'results\\python\\gpu_accuracy.dat'

    c_cpu_times_path = 'results\\c++\\cpu_times.dat'
    c_cpu_accuracy_path = 'results\\c++\\cpu_accuracy.dat'
    c_gpu_times_path = 'results\\c++\\gpu_times.dat'
    c_gpu_accuracy_path = 'results\\c++\\gpu_accuracy.dat'

    #Get time data
    times_paths = [p_cpu_times_path, p_gpu_times_path, c_cpu_times_path, c_gpu_times_path]
    times = []
    for path in times_paths:
        temp_time = collections.defaultdict(list)
        with open(path, 'r') as file:
            for line in file:
                key, value = line.split(',')
                temp_time[int(key)] = float(value)
        temp_time = collections.OrderedDict(sorted(temp_time.items()))
        times.append(temp_time)

    #Get accuracy data
    acc_paths = [p_cpu_accuracy_path, p_gpu_accuracy_path, c_cpu_accuracy_path, c_gpu_accuracy_path]
    accuracies = []
    for path in acc_paths:
        temp_acc = collections.defaultdict(list)
        with open(path, 'r') as file:
            for line in file:
                if line.strip():
                    key, value = line.split(',')
                    temp_acc[int(key)].append(float(value) * 100)
        temp_acc = collections.OrderedDict(sorted(temp_acc.items()))
        accuracies.append(temp_acc)
        
    return times, accuracies

if __name__ == '__main__':
    visualize()
