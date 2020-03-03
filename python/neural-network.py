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
    parser.add_argument('--alpha', type=float, default=0.03, help='learning rate')
    parser.add_argument('--i_alpha', type=float, default=0.03,
                        help='initial learning rate')
    parser.add_argument('--decay', type=float, default=0.02,
                        help='learning rate decay')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--n_x', type=int, default=784, help='number of inputs')
    parser.add_argument('--n_h', type=int, default=400,
                        help='number of hidden units')
    parser.add_argument('--n_o', type=int, default=10,
                        help='number of output units')
    parser.add_argument('--beta', type=float, default=0.9,
                        help='parameter for momentum')
    parser.add_argument('--batch_size', type=int,
                        default=1000, help='input batch size')
    parser.add_argument('--batch_iters', type=int,
                        default=60, help='batch iterations')
    return parser.parse_args()


def init_data():
    rootDir = Path(sys.path[0]).parent
    train_images = str(rootDir) + "\\MNIST data\\train-images.idx3-ubyte"
    train_label = str(rootDir) + "\\MNIST data\\train-labels.idx1-ubyte"
    test_images = str(rootDir) + "\\MNIST data\\t10k-images.idx3-ubyte"
    test_label = str(rootDir) + "\\MNIST data\\t10k-labels.idx1-ubyte"

    train_image_data = idx2numpy.convert_from_file(train_images)
    train_label_data = idx2numpy.convert_from_file(train_label)
    test_image_data = idx2numpy.convert_from_file(test_images)
    test_label_data = idx2numpy.convert_from_file(test_label)

    items, rows, cols = train_image_data.shape
    train_image_data = train_image_data.reshape(items, rows * cols)
    items, rows, cols = test_image_data.shape
    test_image_data = test_image_data.reshape(items, rows * cols)

    return train_image_data, train_label_data, test_image_data, test_label_data

def init_weights(arch):
    weights = {
        "W1" : np.random.randn(arch[0][0], arch[0][1]) / np.sqrt(arch[0][0]),
        "b1" : np.random.randn(1, arch[0][1]) / np.sqrt(arch[0][0]),
        "W2" : np.random.randn(arch[1][0], arch[1][1]) / np.sqrt(arch[1][0]),
        "b2" : np.random.randn(1, arch[1][1]) / np.sqrt(arch[1][1])
        }
    
    return weights

def init_velocities(arch):
    velocities = {
        "W1" : np.zeros((arch[0][0], arch[0][1])),
        "b1" : np.zeros((1, arch[0][1])),
        "W2" : np.zeros((arch[1][0], arch[1][1])),
        "b2" : np.zeros((1, arch[1][1]))
        }

    return velocities
    
def train():
    #Get opts, data, and weights
    train_input, train_target, test_input, test_target = init_data()
    arch = ((opts.n_x, opts.n_h), (opts.n_h, opts.n_o))
    weights = init_weights(arch)
    velocities = init_velocities(arch)
    
    #Train for n epochs
    for j in range(opts.epochs + 1):

        #Get a random mini batch and shuffle up the original data set. Update alpha
        permutation = np.random.permutation(opts.batch_size * opts.batch_iters)
        X_epoch = train_input[permutation]
        y_epoch = train_target[permutation]
        #opts.alpha = opts.i_alpha * (1 / (1 + opts.decay * j))

        for k in range(opts.batch_iters):
            #Move through the data set according to the batch size
            begin = k * opts.batch_size
            end = begin + opts.batch_size

            X = X_epoch[begin:end]
            y = y_epoch[begin:end]
            
            # First, feed forward through the hidden layer
            outputs = feed_forward(X, weights)
            
            # Then, error back propagation from output to input
            output_error, deltas = back_propagation(weights, outputs, X, y)

            """
            # Finally, updating the weights of the network
            weights['W2'] = weights['W2'] - (opts.alpha * deltas['dW2'])
            weights['b2'] = weights['b2'] - (opts.alpha * deltas['db2'])
            weights['W1'] = weights['W1'] - (opts.alpha * deltas['dW1'])
            weights['b1'] = weights['b1'] - (opts.alpha * deltas['db1'])
            """

            #Using velocities for momentum in SGD
            velocities['W2'] = opts.beta * velocities['W2'] + (1 - opts.beta) * deltas['dW2']
            velocities['b2'] = opts.beta * velocities['b2'] + (1 - opts.beta) * deltas['db2']
            velocities['W1'] = opts.beta * velocities['W1'] + (1 - opts.beta) * deltas['dW1']
            velocities['b1'] = opts.beta * velocities['b1'] + (1 - opts.beta) * deltas['db1']

            weights['W2'] = weights['W2'] - opts.alpha * velocities['W2']
            weights['b2'] = weights['b2'] - opts.alpha * velocities['b2']
            weights['W1'] = weights['W1'] - opts.alpha * velocities['W1']
            weights['b1'] = weights['b1'] - opts.alpha * velocities['b1']

            
        # From time to time, reporting the results
        if (j % 5) == 0:
            train_error = np.mean(np.abs(output_error))
            print('Epoch {:5}'.format(j), end=' - ')
            print('error: {:0.4f}'.format(train_error), end= ' - ')

            train_accuracy = accuracy(target=y, predictions=(get_predictions(outputs, y)))
            test_preds = predict(test_input, test_target, weights)
            test_accuracy = accuracy(target=test_target, predictions=test_preds)

            print('acc: train {:0.3f}'.format(train_accuracy), end= ' | ')
            print('test {:0.3f}'.format(test_accuracy))
            

    
def feed_forward(inputs, weights):
    #Empty return list
    outputs = {}
    
    #Dot product of input value and weight
    z1 = np.dot(inputs, weights['W1']) + weights['b1']
    
    #Input is now equal to activation of output
    a1 = sigmoid(z1)

    #Dot product of hidden layer out and weight
    z2 = np.dot(a1, weights['W2']) + weights['b2']

    #Run through softmax
    a2 = softmax(z2)

    outs = {"Z1": z1, "A1": a1, "Z2": z2, "A2": a2}
    return outs

    
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1/(1 + np.exp(-z))


def sigmoid_prime(z):
    return z * (1 - z)

def softmax(z):
   # z = np.clip(z, -709, 709)
    t = np.exp(z)
    a = np.exp(z) / np.sum(t, axis=1).reshape(-1,1)
    return a
        


def back_propagation(weights, outputs, train_input, train_target):
    deltas = {}
    
    output_error = calculate_error(train_target, outputs['A2'])
    error_gradient = error_derivative(train_target, outputs['A2'])
    out_delta = np.dot(outputs['A1'].T, error_gradient) / error_gradient.shape[0]
    prior_error = error_gradient
    deltas['dW2'] = out_delta
    deltas['db2'] = np.sum(error_gradient, axis=0, keepdims=True) / error_gradient.shape[0]

    hidden_out_error = np.dot(error_gradient, weights['W2'].T)
    hidden_error = hidden_out_error * outputs['A1'] * sigmoid_prime(outputs['A1'])
    hidden_delta = np.matmul(train_input.T, hidden_error)
    deltas['dW1'] = hidden_delta
    deltas['db1'] = np.sum(hidden_error, axis=0, keepdims=True) / error_gradient.shape[0]

    return output_error, deltas


    
def calculate_error(target, output):
    #Cost - average loss of each output node
    rows, cols = output.shape
    reshaped_target = np.zeros((rows, 10))
    reshaped_target[np.arange(reshaped_target.shape[0]), target]=1
    ce = -np.sum(reshaped_target * np.log(output + 1e-8))
    return round(ce, 2)

def error_derivative(target, output):
    rows, cols = output.shape
    reshaped_target = np.zeros((rows, 10))
    reshaped_target[np.arange(reshaped_target.shape[0]), target]=1
    return output - reshaped_target

    
def accuracy(target, predictions):
    correct_preds = np.sum(predictions.astype(int))
    return correct_preds / len(target)


def predict(inputs, target, weights):
    outputs = feed_forward(inputs, weights)
    preds = get_predictions(outputs, target=target).astype(int)
    return preds



def get_predictions(outputs, target):
    predicts = np.argmax(outputs['A2'], axis=1)
    return predicts == target


start_time = time.time()
opts = init_params()
train()
print("--- %s seconds ---" % (time.time() - start_time))
