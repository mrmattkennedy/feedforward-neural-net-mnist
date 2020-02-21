from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_prime(s):
    return s * (1 -s)

def init(inp, out):
    #randn creates random element, divide by squareroot of inp for randomness
    return np.random.randn(inp, out) / np.sqrt(inp)

def create_architecture(input_layer, first_layer, output_layer, random_seed=0):
    #Create a random seed
    np.random.seed(random_seed)
    #Size of each layer. X.shape[1] is the number of inputs
    layers = input_layer, first_layer , output_layer
    #Number of input/output for each layer. Takes first num, next num, combines, and continues
    arch = list(zip(layers[:-1], layers[1:]))
    weights = [init(inp, out) for inp, out in arch]
    return weights

def feed_forward(X, weights):
    #Create copy of test data
    a = X.copy()
    #Empty return list
    out = list()
    for W in weights:
        #Dot product of input value and weight
        z = np.dot(a, W)
        #Input is now equal to activation of output
        a = sigmoid(z)
        #Append new input to return
        out.append(a)
        
    return out

"""
L1 - layer 1 values (hidden)
L2 - layer 2 values (output)
weights - current weight values
y - correct responses
"""
def back_propagation(l1, l2, weights, y):
    #Reshape y 
    l2_error = y.reshape(-1, 1) - l2
    l2_delta = l2_error * sigmoid_prime(l2)
    l1_error = l2_delta.dot(weights[1].T)
    l1_delta = l1_error * sigmoid_prime(l1)
    return l2_error, l1_delta, l2_delta

def update_weights(X, l1, l1_delta, l2_delta, weights, alpha=1.0):
    #Learning rate * sum of all costs * inputs for each node
    weights[1] = weights[1] + (alpha * l1.T.dot(l2_delta))
    weights[0] = weights[0] + (alpha * X.T.dot(l1_delta))
    return weights

def accuracy(true_label, predicted):
    correct_preds = np.ravel(predicted)==true_label
    return np.sum(correct_preds) / len(true_label)


def predict(X, weights):
    _, l2 = feed_forward(X, weights)
    preds = np.ravel((l2 > 0.5).astype(int))
    return preds

np.random.seed(0)

coord, cl = make_moons(300, noise=0.05)
X, Xt, y, yt = train_test_split(coord, cl,
                                test_size=0.30, 
                                random_state=0)
"""
X - input training data set
Xt - input test data set
y - output training data set correct responses
yt - output test data set correct responses
"""
weights = create_architecture(X.shape[1], 3, 1)
#print(weights)
l1, l2 = feed_forward(X, weights)
l2_error, l1_delta, l2_delta = back_propagation(l1, l2, weights, y)
test = [[1, 2, 3], [4, 5, 6]]
test = np.array(test)
print(l1.T.shape)
print(l2_delta.shape)
"""
for j in range(30000 + 1):

    # First, feed forward through the hidden layer
    l1, l2 = feed_forward(X, weights)
    
    # Then, error back propagation from output to input
    l2_error, l1_delta, l2_delta = back_propagation(l1, l2, weights, y)
    
    # Finally, updating the weights of the network
    weights = update_weights(X, l1, l1_delta, l2_delta, weights, alpha=0.05)
    
    # From time to time, reporting the results
    if (j % 5000) == 0:
        train_error = np.mean(np.abs(l2_error))
        print('Epoch {:5}'.format(j), end=' - ')
        print('error: {:0.4f}'.format(train_error), end= ' - ')
        
        train_accuracy = accuracy(true_label=y, predicted=(l2 > 0.5))
        test_preds = predict(Xt, weights)
        test_accuracy = accuracy(true_label=yt, predicted=test_preds)
        
        print('acc: train {:0.3f}'.format(train_accuracy), end= ' | ')
        print('test {:0.3f}'.format(test_accuracy))
"""
