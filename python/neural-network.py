import pdb
import sys
import math
import time
import argparse
import idx2numpy
import numpy as np
from pathlib import Path


def init_params():
    parser = argparse.ArgumentParser()

    # hyperparameters setting
    parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train')
    parser.add_argument('--n_x', type=int, default=784, help='number of inputs')
    parser.add_argument('--n_h', type=int, default=64,
                        help='number of hidden units')
    parser.add_argument('--beta', type=float, default=0.9,
                        help='parameter for momentum')
    parser.add_argument('--batch_size', type=int,
                        default=500, help='input batch size')
    return parser.parse_args()

def init_data():
    rootDir = Path(sys.path[0]).parent

    #Load data from idx3 files
    train_images = str(rootDir) + "\\MNIST data\\train-images.idx3-ubyte"
    train_label = str(rootDir) + "\\MNIST data\\train-labels.idx1-ubyte"
    test_images = str(rootDir) + "\\MNIST data\\t10k-images.idx3-ubyte"
    test_label = str(rootDir) + "\\MNIST data\\t10k-labels.idx1-ubyte"

    #Put data to numpy arrays
    x_train = idx2numpy.convert_from_file(train_images)
    x_test = idx2numpy.convert_from_file(train_label)
    y_train = idx2numpy.convert_from_file(test_images)
    y_test = idx2numpy.convert_from_file(test_label)

    
    #Stack for splitting to test/train sets
    X = np.vstack((x_train, y_train))
    y = np.hstack((x_test, y_test))

    # one-hot encoding
    digits = 10
    rows = y.shape[0]
    Y_new = np.zeros((rows, digits))
    Y_new[np.arange(Y_new.shape[0]), y]=1

    #Split the data sets
    m = 60000
    m_test = X.shape[0] - m
    X_train, X_test = X[:m], X[m:]
    Y_train, Y_test = Y_new[:m], Y_new[m:]

   # return X_train, X_test, Y_train, Y_test
    #pdb.set_trace()
    digits = 10
    rows = x_test.shape[0]
    Y_new = np.zeros((rows, digits))
    Y_new[np.arange(Y_new.shape[0]), x_test]=1

    rows = y_test.shape[0]
    Y_new_t = np.zeros((rows, digits))
    Y_new_t[np.arange(Y_new_t.shape[0]), y_test]=1
    return x_train, Y_new, y_train, Y_new_t


def init_weights(opt):
    weights = {"W1": np.random.randn(opt.n_x, opt.n_h) * np.sqrt(1. / opt.n_h),
          "b1": np.random.randn(1, opt.n_h) * np.sqrt(1. / opt.n_h),
          "W2": np.random.randn(opt.n_h, 10) * np.sqrt(1. / 10),
          "b2": np.random.randn(1, 10) * np.sqrt(1. / 10)}
    
    return weights



def train():
    X_train, X_test, Y_train, Y_test = init_data()
    opt = init_params()
    weights = init_weights(opt)
    dW1, db1, dW2, db2 = 0, 0, 0, 0
    for i in range(opt.epochs):

        # shuffle training set
        permutation = np.random.permutation(7000)
        #X_train_shuffled = X_train[permutation]
        #Y_train_shuffled = Y_train[permutation]
        #pdb.set_trace()

        #for j in range(100):
            # get mini-batch
        #begin = j * opt.batch_size
        #end = begin + opt.batch_size
        
        #X = X_train_shuffled[begin:end]
        #pdb.set_trace()
        X = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
        #Y = Y_train_shuffled[begin:end]
        Y = X_test
        #m_batch = end - begin

        # forward and backward
        cache = feed_forward(X, weights)
        grads = back_propagate(X, Y, weights, cache, X_train.shape[0])

        # with momentum (optional)
        """
        dW1 = (opt.beta * dW1 + (1. - opt.beta) * grads["dW1"])
        db1 = (opt.beta * db1 + (1. - opt.beta) * grads["db1"])
        dW2 = (opt.beta * dW2 + (1. - opt.beta) * grads["dW2"])
        db2 = (opt.beta * db2 + (1. - opt.beta) * grads["db2"])
        """
        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]
        #pdb.set_trace()
        # gradient descent
        weights["W1"] = weights["W1"] - opt.lr * dW1
        weights["b1"] = weights["b1"] - opt.lr * db1
        weights["W2"] = weights["W2"] - opt.lr * dW2
        weights["b2"] = weights["b2"] - opt.lr * db2

        # forward pass on training set
        #pdb.set_trace()
        X = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
        cache = feed_forward(X, weights)
        train_loss = compute_loss(X_test, cache["A2"])
        #pdb.set_trace()
        train_accuracy = compute_accuracy(X_test, cache["A2"])

        # forward pass on test set
        #pdb.set_trace()
        X = Y_train.reshape((Y_train.shape[0], Y_train.shape[1] * Y_train.shape[2]))
        cache = feed_forward(X, weights)
        test_loss = compute_loss(Y_test, cache["A2"])
        test_accuracy = compute_accuracy(Y_test, cache["A2"])
        print("Epoch {}: training loss = {}, test loss = {}".format(
            i + 1, round(train_loss, 2), round(test_loss, 2)))
        print("Training accuracy = {}, Test accuracy = {}".format(
            round(train_accuracy, 2), round(test_accuracy, 2)))
        
def feed_forward(X, weights):
    """
    feed forward network: 2 - layer neural net

    inputs:
        params: dictionay a dictionary contains all the weights and biases

    return:
        cache: dictionay a dictionary contains all the fully connected units and activations
    """
    cache = {}
    
    # Z1 = W1.dot(x) + b1
    cache["Z1"] = np.dot(X, weights["W1"]) + weights["b1"]
    # A1 = sigmoid(Z1)
    cache["A1"] = sigmoid(cache["Z1"])

    # Z2 = W2.dot(A1) + b2
    cache["Z2"] = np.dot(cache["A1"], weights["W2"]) + weights["b2"]

    # A2 = softmax(Z2)
    cache["A2"] = np.exp(cache["Z2"]) / np.sum(np.exp(cache["Z2"]), axis=0)
    
    return cache


def sigmoid(z):
    e_max = 709
    z = np.where(z > e_max, e_max, z)
    z = np.where(z < -e_max, -e_max, z)
    return 1/(1 + np.exp(-z))

def sigmoid_prime(s):
    return s * (1 -s) 

        

def back_propagate(X, Y, weights, cache, batch_size):
    """
    back propagation

    inputs:
        params: dictionay a dictionary contains all the weights and biases
        cache: dictionay a dictionary contains all the fully connected units and activations

    return:
        grads: dictionay a dictionary contains the gradients of corresponding weights and biases
    """
    #pdb.set_trace()
    # error at last layer
    dZ2 = cache["A2"] - Y

    # gradients at last layer (Py2 need 1. to transform to float)
    dW2 = (1. / batch_size) * np.dot(cache["A1"].T, dZ2)
    db2 = (1. / batch_size) * np.sum(dZ2, axis=0, keepdims=True)

    # back propgate through first layer
    dA1 = np.dot(dZ2, weights["W2"].T)
    dZ1 = dA1 * cache['A1'] * sigmoid_prime(cache['A1'])

    # gradients at first layer (Py2 need 1. to transform to float)
    dW1 = (1. / batch_size) * np.dot(dZ1.T, X).T
    db1 = (1. / batch_size) * np.sum(dZ1, axis=0, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads


    
def compute_loss(Y, Y_hat):
    """
    compute loss function
    """
    #pdb.set_trace()
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1./m) * L_sum

    return L

def compute_accuracy(Y, Y_hat):
    #Get the correct predictions, then compare to target.)
    Y_hat = np.argmax(Y_hat, axis=1)
    Y = np.argmax(Y, axis=1)
    correct_preds = Y_hat == Y
    correct_preds = np.sum(correct_preds.astype(int))
    return correct_preds / len(Y)

# parse the arguments
start_time = time.time()
train()
print("--- %s seconds ---" % (time.time() - start_time))
