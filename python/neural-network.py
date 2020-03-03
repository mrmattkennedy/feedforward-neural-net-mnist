import pdb
import sys
import math
import time
import traceback
import idx2numpy
import numpy as np
from pathlib import Path
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def train():
    weights = init_weights()
    for j in range(epochs + 1):
        # First, feed forward through the hidden layer
        outputs = feed_forward(train_input, weights)
        
        # Then, error back propagation from output to input
        output_error, deltas = back_propagation(weights, outputs)

        # Finally, updating the weights of the network
        weights['W2'] = weights['W2'] - (alpha * deltas['dW2'])
        weights['b2'] = weights['b2'] - (alpha * deltas['db2'])
        weights['W1'] = weights['W1'] - (alpha * deltas['dW1'])
        weights['b1'] = weights['b1'] - (alpha * deltas['db1'])

        # From time to time, reporting the results
        if (j % 1) == 0:
            train_error = np.mean(np.abs(output_error))
            print('Epoch {:5}'.format(j), end=' - ')
            print('error: {:0.4f}'.format(train_error), end= ' - ')

            train_accuracy = accuracy(target=train_target, predictions=(get_predictions(outputs, train_target)))
            test_preds = predict(test_input, weights)
            test_accuracy = accuracy(target=test_target, predictions=test_preds)

            print('acc: train {:0.3f}'.format(train_accuracy), end= ' | ')
            print('test {:0.3f}'.format(test_accuracy))


def init_weights():
    arch = ((rows * cols, 500), (500, 10))
    weights = {
        "W1" : np.random.randn(arch[0][0], arch[0][1]) / np.sqrt(arch[0][0]),
        "b1" : np.random.randn(1, arch[0][1]) / np.sqrt(arch[0][0]),
        "W2" : np.random.randn(arch[1][0], arch[1][1]) / np.sqrt(arch[1][0]),
        "b2" : np.random.randn(1, arch[1][1]) / np.sqrt(arch[1][1])}
    
    return weights


def create_architecture(self, in_layer, out_layer, hidden_layers=None, random_seed=0):
    """
    Creates the architecture for the network.
    Sets the random seed for numpy, then gets the sizes for each layer.
    Next, creates a temporary architecture list, with the current
    and next list sizes as each element.
    Last, initializes random weights of size inp, out.

    Parameters
    ----------
    in_layer : int
        Size of the input layer
    out_layer : int
        Size of the output layer
    hidden_layers : tuple(ints)
        Size of the hidden layers, if any
    random_seed : int
        Random seed for numpy
    """
    np.random.seed(random_seed)
    #Size of each layer. X.shape[1] is the number of inputs
    layers = tuple((in_layer,))
    if hidden_layers is not None:
        for size in hidden_layers:
            layers += tuple((size,))
    layers += tuple((out_layer,))
    
    #Number of input/output for each layer. Takes first num, next num, combines, and continues
    arch = list(zip(layers[:-1], layers[1:]))
    #Create list of weights
    #self.weights = [self.init_weights(inp, out) for inp, out in arch]
    self.weights = self.init_weights(arch)   

    
def feed_forward(inputs, weights):
    #pdb.set_trace()
    #Empty return list
    outputs = {}
    
    #Dot product of input value and weight
    z1 = np.dot(inputs, weights['W1']) + weights['b1']
    
    #Input is now equal to activation of output
    a1 = activation_func(z1, 'sigmoid')

    #Dot product of hidden layer out and weight
    z2 = np.dot(a1, weights['W2']) + weights['b2']

    #Run through softmax
    a2 = activation_func(z2, 'softmax')

    
    outs = {"Z1": z1, "A1": a1, "Z2": z2, "A2": a2}
    return outs

    
def activation_func(input_values, name='sigmoid'):
    """
    Runs value through the activation function for a neuron.
    Defaults to sigmoid function.

    Parameters
    ----------
    name : string
        Name of the activation function.
    input_values : np.array
        Current input values in the network.

    Returns
    -------
    np.array output_activated
        The array of inputs ran through the activation
        function as a list.
    """

    if name == 'sigmoid':
        #input_values = np.where(input_values > e_max, e_max, input_values)
        #input_values = np.where(input_values < -e_max, -e_max, input_values)
        return 1/(1 + np.exp(-input_values))
    elif name == 'tanh':
        return np.tanh(input_values)
    elif name == 'softmax':
        t = np.exp(input_values)
        a = np.exp(input_values) / np.sum(t, axis=1).reshape(-1,1)
        return a


def activation_func_prime(input_values, name='sigmoid'):
    if name == 'sigmoid':
        return input_values * (1 - input_values)
    elif name == 'softmax':
        #Get the diagonal
        rows, cols = input_values.shape            
        out = np.zeros((rows, cols, cols))
        out[:, np.arange(cols), np.arange(cols)] = input_values

        #Get the dot product of 2nd and 3rd axes
        vals = np.reshape(input_values, (rows, cols, 1))
        vals_T = vals.reshape((rows, 1, cols))
        prod = np.array([np.dot(vals[row], vals_T[row]) for row in range(rows)])
                        
        return out - prod

        

def back_propagation(weights, outputs):
    deltas = {}
    
    output_error = calculate_error(train_target, outputs['A2'])
    error_gradient = error_derivative(train_target, outputs['A2'])
    out_delta = np.dot(outputs['A1'].T, error_gradient) / error_gradient.shape[0]
    prior_error = error_gradient
    deltas['dW2'] = out_delta
    deltas['db2'] = np.sum(error_gradient, axis=0, keepdims=True) / error_gradient.shape[0]

    hidden_out_error = np.dot(error_gradient, weights['W2'].T)
    hidden_error = hidden_out_error * outputs['A1'] * activation_func_prime(outputs['A1'])
    hidden_delta = np.matmul(train_input.T, hidden_error)
    deltas['dW1'] = hidden_delta
    deltas['db1'] = np.sum(hidden_error, axis=0, keepdims=True) / error_gradient.shape[0]

    return output_error, deltas
    
def calculate_error(target, output):
    rows, cols = output.shape
    reshaped_target = np.zeros((rows, 10))
    reshaped_target[np.arange(reshaped_target.shape[0]), target]=1
    #Cost - average loss of each output node
    ce = -np.mean(np.sum(reshaped_target * np.log(output + 1e-8), axis=1))
    return ce

def error_derivative(target, output):
    rows, cols = output.shape
    reshaped_target = np.zeros((rows, 10))
    reshaped_target[np.arange(reshaped_target.shape[0]), target]=1
    return output - reshaped_target

    
def accuracy(target, predictions):
    correct_preds = np.sum(predictions.astype(int))
    return correct_preds / len(target)



def predict(inputs, weights):
    outputs = feed_forward(inputs, weights)
    preds = get_predictions(outputs, target=test_target).astype(int)
    return preds



def get_predictions(outputs, target):
    predicts = np.argmax(outputs['A2'], axis=1)
    return predicts == target

np.random.seed(0)
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
_, in_nodes = train_image_data.shape

start_time = time.time()
alpha = 0.05
epochs = 30
e_max = 709
train_input=train_image_data
train_target=train_label_data
test_input=test_image_data
test_target=test_label_data
"""
nn = neural_network(in_nodes, 10,
                out_func='softmax',
                hl_sizes=500,
                hl_functions='sigmoid',
                alpha=0.05,
                epochs=30, bias=False)

nn.train(train_input=train_image_data,
         train_target=train_label_data,
         test_input=test_image_data,
         test_target=test_label_data)
"""
train()
print("--- %s seconds ---" % (time.time() - start_time))
