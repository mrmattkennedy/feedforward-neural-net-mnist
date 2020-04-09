import pdb
import sys
import math
import time
import argparse
import traceback
import idx2numpy
import numpy as np
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
    parser.add_argument('--batch_size', type=int, default=60000,
                        help='input batch size')
    parser.add_argument('--batches', type=int, default=1,
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

    return train_data, label_data

def init_weights(arch):
    weights = {
        "W1" : np.random.randn(arch[0][0], arch[0][1]) * np.sqrt(1 / arch[0][0]),
        "b1" : np.random.randn(1, arch[0][1]) * np.sqrt(1 / arch[0][0]),
        "W2" : np.random.randn(arch[1][0], arch[1][1]) * np.sqrt(1 / arch[1][0]),
        "b2" : np.random.randn(1, arch[1][1]) * np.sqrt(1 / arch[1][1]),
        "W3" : np.random.randn(arch[2][0], arch[2][1]) * np.sqrt(1 / arch[2][0]),
        "b3" : np.random.randn(1, arch[2][1]) * np.sqrt(1 / arch[2][1])
        }
    
    return weights


def init_velocities(arch):
    velocities = {
        "W1" : np.zeros((arch[0][0], arch[0][1])),
        "b1" : np.zeros((1, arch[0][1])),
        "W2" : np.zeros((arch[1][0], arch[1][1])),
        "b2" : np.zeros((1, arch[1][1])),
        "W3" : np.zeros((arch[2][0], arch[2][1])),
        "b3" : np.zeros((1, arch[2][1]))
        }

    return velocities

    
def train():
    #Get opts, data, weights, velocities
    X, y = init_data()
    arch = ((opts.n_x, opts.n_h), (opts.n_h, opts.n_h2), (opts.n_h2, opts.n_o))
    weights = init_weights(arch)
    velocities = init_velocities(arch)
    opts.alpha = 0.002
    
    start_time = time.time()
    accuracies = []
    #Train for n epochs
    for j in range(opts.epochs):
        
        #Shuffle data
        permutation = np.random.permutation(X.shape[0])
        X = X[permutation]
        y = y[permutation]
        
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

            #pdb.set_trace()    
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

            outputs = feed_forward(X_train, weights)
            train_accuracy = accuracy(target=y_train, predictions=(get_predictions(outputs, y_train)))
            test_preds = predict(X_test, y_test, weights)
            test_accuracy = accuracy(target=y_test, predictions=test_preds)

            print('acc: train {:0.6f}'.format(train_accuracy), end= ' | ')
            print('test {:0.6f}'.format(test_accuracy))
            accuracies.append(test_accuracy)
            
    return time.time() - start_time, accuracies
    
def feed_forward(inputs, weights):
    #Empty return dict
    outputs = {}
    
    #Dot product of input value and weight
    z1 = np.dot(inputs, weights['W1'])# + weights['b1']
    
    #Input is now equal to activation of output
    a1 = sigmoid(z1)

    #Dot product of input value and weight
    z2 = np.dot(a1, weights['W2'])# + weights['b2']
    
    #Input is now equal to activation of output
    a2 = sigmoid(z2)
    
    #Dot product of hidden layer out and weight
    z3 = np.dot(a2, weights['W3'])# + weights['b3']

    #Run through softmax
    a3 = softmax(z3)

    outs = {"Z1": z1, "A1": a1, "Z2": z2, "A2": a2, "Z3": z3, "A3": a3}
    return outs

    
def sigmoid(z):
    z = np.clip(z, -500, 500)
    denom = 1 + np.exp(-z)
    return 1/denom

def sigmoid_prime(z):
    return z * (1 - z)
    
def softmax(z):
    t = np.exp(z)
    t_sum = np.sum(t, axis=1).reshape(-1, 1)
    a = t / t_sum
    return a


def back_propagation(weights, outputs, train_input, train_target):
    deltas = {}

    #Calculate the error for the output layer
    output_error = calculate_error(train_target, outputs['A3'])

    #Calculate the error derivative for softmax
    error_gradient = error_derivative(train_target, outputs['A3'])

    #Output delta (gradient) is error derivative * hidden layer outs (average for batch)
    out_delta = np.dot(outputs['A2'].T, error_gradient) / error_gradient.shape[0]

    #Append the delta
    deltas['dW3'] = out_delta

    #Append the bias
    deltas['db3'] = np.sum(error_gradient, axis=0, keepdims=True) / error_gradient.shape[0]

    #Get error for the hidden layer output(previous layer error * weights)
    hidden_out_error_2 = np.dot(error_gradient, weights['W3'].T)

    #Hidden layer error is output error * outputs * sigmoid prime
    hidden_error_2 = hidden_out_error_2 * outputs['A2'] * sigmoid_prime(outputs['A2'])

    #Delta is input * error
    hidden_delta_2 = np.matmul(outputs['A1'].T, hidden_error_2)

    #Append the delta
    deltas['dW2'] = hidden_delta_2

    #Append the bias
    deltas['db2'] = np.sum(hidden_error_2, axis=0, keepdims=True) / error_gradient.shape[0]

    #Get error for the hidden layer output(previous layer error * weights)
    hidden_out_error = np.dot(hidden_error_2, weights['W2'].T)

    #Hidden layer error is output error * outputs * sigmoid prime
    hidden_error = hidden_out_error * outputs['A1'] * sigmoid_prime(outputs['A1'])

    #Delta is input * error
    hidden_delta = np.matmul(train_input.T, hidden_error)

    #Append the delta
    deltas['dW1'] = hidden_delta

    #Append the bias
    deltas['db1'] = np.sum(hidden_error, axis=0, keepdims=True) / error_gradient.shape[0]
    
    #Return
    return output_error, deltas


    
def calculate_error(target, output):
    #Get the shape of the output
    rows, cols = output.shape

    #Reshape from from just a # to all 0's
    reshaped_target = np.zeros((rows, opts.n_o))

    #Change index of correct predictions to a 1
    reshaped_target[np.arange(reshaped_target.shape[0]), target]=1

    #Add up the error
    ce = -np.sum(reshaped_target * np.log(output + 1e-8))

    #Round and return
    return round(ce, 2)



def error_derivative(target, output):
    rows, cols = output.shape
    reshaped_target = np.zeros((rows, opts.n_o))
    reshaped_target[np.arange(reshaped_target.shape[0]), target]=1
    return output - reshaped_target


    
def accuracy(target, predictions):
    #See the total sum of 1's (True's where predictions matched target)
    correct_preds = np.sum(predictions.astype(int))

    #Return correct / total
    return correct_preds / len(target)



def predict(inputs, target, weights):
    #Feed forward test inputs
    outputs = feed_forward(inputs, weights)

    #Get the predictions in a usable format
    preds = get_predictions(outputs, target=target).astype(int)

    #Return preds
    return preds



def get_predictions(outputs, target):
    #For each row, get the predictions (where the 1 is)
    predicts = np.argmax(outputs['A3'], axis=1)

    #Return where predictions match target
    return predicts == target



def save_results():
    train_size = 60000    
    batch_sizes = [int(line.rstrip('\n')) for line in open('data/batch_sizes.data', 'r').readline().split(', ')]
    batch_sizes = [item for item in batch_sizes if item >= 30]
    
    times = {}
    accuracies = {}
        
    for size in batch_sizes:    
        opts.batch_size = size
        opts.batches = int(train_size / opts.batch_size)
        total_time, results = train()

        times[size] = total_time
        accuracies[size] = results

    cpu_times_path = 'data\\cpu_times.dat'
    cpu_accuracy_path = 'data\\cpu_accuracy.dat'

    with open(cpu_times_path, 'w') as file:
        for key, value in times.items():
            line = str(key) + ',' + str(value)
            file.write(line + '\n')
    
    with open(cpu_accuracy_path, 'w') as file:
        for key, value in accuracies.items():
            for item in value:
                line = str(key) + ',' + str(item)
                file.write(line + '\n')
            file.write('\n')
            
opts = init_params()
if __name__ == '__main__':
    #save_results()
    train()

