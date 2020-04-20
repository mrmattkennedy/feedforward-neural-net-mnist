import pdb
import sys
import math
import time
import argparse
import traceback
import idx2numpy
import numpy as np
import skcuda.misc as misc
import skcuda.linalg as linalg
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from pathlib import Path


def init_params():
    parser = argparse.ArgumentParser()

    # hyperparameters setting
    parser.add_argument('--alpha', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--decay', type=float, default=0.0001,
                        help='learning rate decay')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train')
    parser.add_argument('--n_x', type=int, default=784,
                        help='number of inputs')
    parser.add_argument('--n_h', type=int, default=600,
                        help='number of hidden units')
    parser.add_argument('--n_h2', type=int, default=500,
                        help='number of hidden units')
    parser.add_argument('--n_o', type=int, default=10,
                        help='number of output units')
    parser.add_argument('--beta', type=float, default=0.9,
                        help='parameter for momentum')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='input batch size')
    parser.add_argument('--batches', type=int, default=6,
                        help='batch iterations')
    return parser.parse_args()


def init_data():
    #Get file paths
    rootDir = Path(sys.path[0]).parent
    train_images = str(rootDir) + "\\MNIST data\\train-images.idx3-ubyte"
    train_label = str(rootDir) + "\\MNIST data\\train-labels.idx1-ubyte"
    test_images = str(rootDir) + "\\MNIST data\\t10k-images.idx3-ubyte"
    test_label = str(rootDir) + "\\MNIST data\\t10k-labels.idx1-ubyte"

    #Read in from file to numpy array
    train_image_data = idx2numpy.convert_from_file(train_images)
    train_label_data = idx2numpy.convert_from_file(train_label)
    test_image_data = idx2numpy.convert_from_file(test_images)
    test_label_data = idx2numpy.convert_from_file(test_label)

    #Reshaped the inputs from 3D to 2D
    items, rows, cols = train_image_data.shape
    train_image_data = train_image_data.reshape(items, rows * cols)
    items, rows, cols = test_image_data.shape
    test_image_data = test_image_data.reshape(items, rows * cols)

    train_data = np.vstack((train_image_data, test_image_data))
    label_data = np.hstack((train_label_data, test_label_data))

    return gpuarray.to_gpu(train_data.astype(np.float32)), gpuarray.to_gpu(label_data.astype(np.float32))

def init_weights(arch):
    weights = {
        "W1" : gpuarray.to_gpu(np.asarray(np.random.randn(arch[0][0], arch[0][1]) * np.sqrt(1 / arch[0][0]), np.float32)),
        "b1" : np.random.randn(1, arch[0][1]) * np.sqrt(1 / arch[0][0]),
        "W2" : gpuarray.to_gpu(np.asarray(np.random.randn(arch[1][0], arch[1][1]) * np.sqrt(1 / arch[1][0]), np.float32)),
        "b2" : np.random.randn(1, arch[1][1]) * np.sqrt(1 / arch[1][1]),
        "W3" : gpuarray.to_gpu(np.asarray(np.random.randn(arch[2][0], arch[2][1]) * np.sqrt(1 / arch[2][0]), np.float32)),
        "b3" : np.random.randn(1, arch[2][1]) * np.sqrt(1 / arch[2][1])
        }

    #Reshape bias weights
    weights['b1'] = weights['b1'].reshape(weights['b1'].shape[1])
    weights['b2'] = weights['b2'].reshape(weights['b2'].shape[1])
    weights['b3'] = weights['b3'].reshape(weights['b3'].shape[1])
    
    weights['b1'] = gpuarray.to_gpu(np.asarray(np.tile(weights['b1'], (opts.batch_size, 1)), np.float32))
    weights['b2'] = gpuarray.to_gpu(np.asarray(np.tile(weights['b2'], (opts.batch_size, 1)), np.float32))
    weights['b3'] = gpuarray.to_gpu(np.asarray(np.tile(weights['b3'], (opts.batch_size, 1)), np.float32))
    return weights


def init_velocities(arch):
    velocities = {
        "W1" : gpuarray.to_gpu(np.asarray(np.zeros((arch[0][0], arch[0][1])), np.float32)),
        "b1" : np.zeros((1, arch[0][1])),
        "W2" : gpuarray.to_gpu(np.asarray(np.zeros((arch[1][0], arch[1][1])), np.float32)),
        "b2" : np.zeros((1, arch[1][1])),
        "W3" : gpuarray.to_gpu(np.asarray(np.zeros((arch[2][0], arch[2][1])), np.float32)),
        "b3" : np.zeros((1, arch[2][1]))
        }

    #Reshape bias weights
    velocities['b1'] = velocities['b1'].reshape(velocities['b1'].shape[1])
    velocities['b2'] = velocities['b2'].reshape(velocities['b2'].shape[1])
    velocities['b3'] = velocities['b3'].reshape(velocities['b3'].shape[1])
    
    velocities['b1'] = gpuarray.to_gpu(np.asarray(np.tile(velocities['b1'], (opts.batch_size, 1)), np.float32))
    velocities['b2'] = gpuarray.to_gpu(np.asarray(np.tile(velocities['b2'], (opts.batch_size, 1)), np.float32))
    velocities['b3'] = gpuarray.to_gpu(np.asarray(np.tile(velocities['b3'], (opts.batch_size, 1)), np.float32))
    return velocities

    
def train():
    #Get opts, data, weights, velocities
    linalg.init()
    X, y = init_data()
    arch = ((opts.n_x, opts.n_h), (opts.n_h, opts.n_h2), (opts.n_h2, opts.n_o))
    weights = init_weights(arch)
    velocities = init_velocities(arch)
    opts.alpha = 0.002
    
    epoch_accuracies = []
    start_time = time.time()
    #Train for n epochs
    for j in range(opts.epochs):            
        
        #Shuffle data
        permutation = np.random.permutation(X.shape[0])
        X = gpuarray.to_gpu(X.get()[permutation])
        y = gpuarray.to_gpu(y.get()[permutation])
        
        #Get the train and test data
        X_train = X[:opts.batch_size * opts.batches]
        y_train = y[:opts.batch_size * opts.batches]
        X_test = X[opts.batch_size * opts.batches:]
        y_test = y[opts.batch_size * opts.batches:]
        opts.alpha *= (1 / (1 + opts.decay * j))

        for k in range(opts.batches):
            #Move through the data set according to the batch size
            begin = k * opts.batch_size
            end = begin + opts.batch_size

            X_batch = X_train[begin:end]
            y_batch = y_train[begin:end]
            
            # Feed forward
            outputs = feed_forward(X_batch, weights)
            
            # Backpropagate, get error as well            
            output_error, deltas = back_propagation(weights, outputs, X_batch, y_batch)
            
            #Using velocities for momentum in SGD
            velocities['W3'] = opts.beta * velocities['W3'] + (1 - opts.beta) * deltas['dW3']
            velocities['b3'] = opts.beta * velocities['b3'] + (1 - opts.beta) * deltas['db3']
            velocities['W2'] = opts.beta * velocities['W2'] + (1 - opts.beta) * deltas['dW2']
            velocities['b2'] = opts.beta * velocities['b2'] + (1 - opts.beta) * deltas['db2']
            velocities['W1'] = opts.beta * velocities['W1'] + (1 - opts.beta) * deltas['dW1']
            velocities['b1'] = opts.beta * velocities['b1'] + (1 - opts.beta) * deltas['db1']
    
            #Update weights
            weights['W3'] = weights['W3'] - opts.alpha * velocities['W3']
            weights['b3'] = weights['b3'] - opts.alpha * velocities['b3']
            weights['W2'] = weights['W2'] - opts.alpha * velocities['W2']
            weights['b2'] = weights['b2'] - opts.alpha * velocities['b2']
            weights['W1'] = weights['W1'] - opts.alpha * velocities['W1']
            weights['b1'] = weights['b1'] - opts.alpha * velocities['b1']
    
        # From time to time, reporting the results
        if (j % 1) == 0:
            train_error = np.mean(np.abs(output_error))
            print('Epoch {:5}'.format(j), end=' - ')
            print('loss: {:0.6f}'.format(train_error), end= ' - ')

            #Get measurements
            outputs = feed_forward(X_train[:opts.batch_size], weights)
            train_accuracy = accuracy(target=y_train[:opts.batch_size], predictions=(get_predictions(outputs, y_train[:opts.batch_size])))
            test_preds = predict(X_test, y_test, weights)
            test_accuracy = accuracy(target=y_test, predictions=test_preds)

            #Display measurements
            print('acc: train {:0.6f}'.format(train_accuracy), end= ' | ')
            print('test {:0.6f}'.format(test_accuracy))
            
            epoch_accuracies.append(test_accuracy)

    return (time.time() - start_time), epoch_accuracies
    
def feed_forward(inputs, weights):
    #Empty return dict
    outputs = {}

    #Dot product of input value and weight
    z1 = linalg.dot(inputs, weights['W1']) + weights['b1']

    #Input is now equal to activation of output
    a1 = sigmoid(z1)

    #Dot product of input value and weight
    z2 = linalg.dot(a1, weights['W2']) + weights['b2']
    
    #Input is now equal to activation of output
    a2 = sigmoid(z2)
    
    #Dot product of hidden layer out and weight
    z3 = linalg.dot(a2, weights['W3']) + weights['b3']

    #Run through softmax
    a3 = softmax(z3)

    outs = {"Z1": z1, "A1": a1, "Z2": z2, "A2": a2, "Z3": z3, "A3": a3}
    return outs

    
def sigmoid(z):
    #Reduce with factor?
    z_max = gpuarray.max(z).get()
    z_min = gpuarray.min(z).get()
    largest_value = max(z_max, abs(z_min))
    factor = 88 / largest_value #float32 precision requires max value for exp to be 88
    z = (z * factor).astype(np.float32)
    return 1 / (1 + cumath.exp(-z))

def sigmoid_prime(z):
    return z * (1 - z)
    
def softmax(z):
    t = cumath.exp(z)
    t_sum = misc.sum(t, axis=1)
    #Convert gpuarray to nparray, repeat each item in the row to match shape of t, then convert back to gpuarray
    a = t / gpuarray.to_gpu(np.repeat(t_sum.get(), t.shape[1]).reshape((t.shape[0], t.shape[1])))
    return  a


def back_propagation(weights, outputs, train_input, train_target):
    deltas = {}

    #Calculate the error for the output layer
    output_error = calculate_error(train_target, outputs['A3'])

    #Calculate the error derivative for softmax
    error_gradient = error_derivative(train_target, outputs['A3'])

    #Output delta (gradient) is error derivative * hidden layer outs (average for batch)
    out_delta = linalg.dot(linalg.transpose(outputs['A2']), error_gradient) / error_gradient.shape[0]

    #Append the delta
    deltas['dW3'] = out_delta

    #Append the bias
    deltas['db3'] = misc.sum(error_gradient, axis=0) / error_gradient.shape[0]

    #Get error for the hidden layer output(previous layer error * weights)
    hidden_out_error_2 = linalg.dot(error_gradient, linalg.transpose(weights['W3']))

    #Hidden layer error is output error * outputs * sigmoid prime
    hidden_error_2 = hidden_out_error_2 * outputs['A2'] * sigmoid_prime(outputs['A2'])

    #Delta is input * error
    hidden_delta_2 = linalg.dot(linalg.transpose(outputs['A1']), hidden_error_2)

    #Append the delta
    deltas['dW2'] = hidden_delta_2

    #Append the bias
    deltas['db2'] = misc.sum(hidden_error_2, axis=0) / error_gradient.shape[0]

    #Get error for the hidden layer output(previous layer error * weights)
    hidden_out_error = linalg.dot(hidden_error_2, linalg.transpose(weights['W2']))

    #Hidden layer error is output error * outputs * sigmoid prime
    hidden_error = hidden_out_error * outputs['A1'] * sigmoid_prime(outputs['A1'])

    #Delta is input * error
    hidden_delta = linalg.dot(linalg.transpose(train_input), hidden_error)

    #Append the delta
    deltas['dW1'] = hidden_delta

    #Append the bias
    deltas['db1'] = misc.sum(hidden_error, axis=0) / error_gradient.shape[0]

    #Reshape bias deltas
    
    deltas['db1'] = gpuarray.to_gpu(np.asarray(np.tile(deltas['db1'].get(), (opts.batch_size, 1)), np.float32))
    deltas['db2'] = gpuarray.to_gpu(np.asarray(np.tile(deltas['db2'].get(), (opts.batch_size, 1)), np.float32))
    deltas['db3'] = gpuarray.to_gpu(np.asarray(np.tile(deltas['db3'].get(), (opts.batch_size, 1)), np.float32))
    
    #Return
    return output_error, deltas


    
def calculate_error(target, output):
    #Get the shape of the output
    rows, cols = output.shape

    #Reshape from from just a # to all 0's
    reshaped_target = np.zeros((rows, opts.n_o))

    #Change index of correct predictions to a 1
    reshaped_target[np.arange(reshaped_target.shape[0]), target.get().astype(np.int32)]=1
    reshaped_target = gpuarray.to_gpu(reshaped_target.astype(np.float32))
    
    #Add up the error
    ce = gpuarray.sum(reshaped_target * cumath.log(output + 1e-10)).get()

    #Round and return
    return round(abs(ce), 2)



def error_derivative(target, output):
    rows, cols = output.shape
    reshaped_target = np.zeros((rows, opts.n_o))
    reshaped_target[np.arange(reshaped_target.shape[0]), target.get().astype(np.int32)]=1
    reshaped_target = gpuarray.to_gpu(reshaped_target.astype(np.float32))
    return output - reshaped_target


    
def accuracy(target, predictions):
    #See the total sum of 1's (True's where predictions matched target)
    correct_preds = np.sum(predictions.astype(int))

    #Return correct / total
    return correct_preds / len(target)



def predict(inputs, target, weights):
    #Reshape bias weights
    weights['b1'] = gpuarray.to_gpu(np.asarray(np.tile(weights['b1'].get()[0], (inputs.shape[0], 1)), np.float32))
    weights['b2'] = gpuarray.to_gpu(np.asarray(np.tile(weights['b2'].get()[0], (inputs.shape[0], 1)), np.float32))
    weights['b3'] = gpuarray.to_gpu(np.asarray(np.tile(weights['b3'].get()[0], (inputs.shape[0], 1)), np.float32))
    
    #Feed forward test inputs
    outputs = feed_forward(inputs, weights)

    #Get the predictions in a usable format
    preds = get_predictions(outputs, target=target).astype(int)

    #Reshape bias weights back
    weights['b1'] = gpuarray.to_gpu(np.asarray(np.tile(weights['b1'].get()[0], (opts.batch_size, 1)), np.float32))
    weights['b2'] = gpuarray.to_gpu(np.asarray(np.tile(weights['b2'].get()[0], (opts.batch_size, 1)), np.float32))
    weights['b3'] = gpuarray.to_gpu(np.asarray(np.tile(weights['b3'].get()[0], (opts.batch_size, 1)), np.float32))
    #Return preds
    return preds



def get_predictions(outputs, target):
    #For each row, get the predictions (where the 1 is)
    predicts = np.argmax(outputs['A3'].get(), axis=1)

    #Return where predictions match target
    return predicts == target.get()



def save_results():
    #First iteration using cuda initalizes everything, extremely slow, skews results.
    #Run one epoch just to initialize, then run tests.
    train_size = 60000
    opts.batch_size = train_size
    opts.batches = 1
    opts.epochs = 1
    train()
    opts.epochs = 50
    
    batch_sizes = [int(item.rstrip('\n')) for item in open('..\\results\\batch_sizes.dat', 'r').readline().split(', ')]
    batch_sizes = [item for item in batch_sizes if item >= 30]
    print(batch_sizes)
    exit()
    
    times = {}
    accuracies = {}
        
    for size in batch_sizes:    
        opts.batch_size = size
        opts.batches = int(train_size / opts.batch_size)
        total_time, results = train()

        times[size] = total_time
        accuracies[size] = results

    gpu_times_path = '..\\results\\python\\gpu_times.dat'
    gpu_accuracy_path = '..\\results\\python\\gpu_accuracy.dat'

    with open(gpu_times_path, 'w') as file:
        for key, value in times.items():
            line = str(key) + ',' + str(value)
            file.write(line + '\n')
    
    with open(gpu_accuracy_path, 'w') as file:
        for key, value in accuracies.items():
            for item in value:
                line = str(key) + ',' + str(item)
                file.write(line + '\n')
            file.write('\n')
            
opts = init_params()
if __name__ == '__main__':
    save_results()
