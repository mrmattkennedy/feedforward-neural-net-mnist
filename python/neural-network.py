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


class neural_network:
    """
    Neural Network class

    Instance variables
    ------------------
    Input size : (self.in_size)
        int, size of the input layer
    Output size : (self.out_size)
        int, size of the output layer
    Weights : (self.weights)
        list(floats), all weights for each layer.
    Learning rate : (self.alpha)
        float, learning rate
    Epochs : (self.epochs)
        int, epochs
    Threshold : (self.threshold)
        float, acceptance threshold
    Hidden layer sizes : (self.hl_sizes)
        tuple(int) : sizes for each hidden layer
    Activation functions (self.a_f)
        list(str), list of strings of activation functions, with output last
        
    Constructor parameters
    -----------
    input nodes : in_size
        int, number of input nodes
    output nodes : out_size
        int, number of output nodes
    alpha : alpha
        int, learning rate (default is 0.05)
    epochs : epochs
        int, number of epochs
    threshold : threshold
        float, acceptance threshold
    output function : out_func
        str, name of the output function (default is softmax)
    hidden layers sizes: hl_sizes
        tuple(int), hidden layer sizes (can be None)
    hidden layer functions : hl_functions
        tuple(str), activation function for each hidden layer (ignored if h_l_count is None)

    Global variables
    ----------------
    Function keywords : funcs
        List(str), allowed function names
    """
    funcs = ['sigmoid', 'tanh', 'softmax', '']


    
    def __init__(self, in_size, out_size,
                 alpha=0.05, epochs=30000, threshold=0.5, bias=True,
                 out_func=None, hl_sizes=None, hl_functions=None):
        
        #Verify sizes are numbers above 0
        assert type(in_size) is int, "Size of input layer needs to be an int"
        assert type(out_size) is int, "Size of output layer needs to be an int"
        assert in_size >= 1, "Size of input layer must be a positive integer"
        assert out_size >= 1, "Size of output layer must be a positive integer"
        assert out_func is None or out_func in neural_network.funcs, "Function must be one of the following: " + ", ".join(neural_network.funcs)
        
        #Make sure sizes are provided if functions are
        assert not (hl_sizes is None and hl_functions is not None), "Must have sizes with functions"
        if (type(hl_sizes) is not int and type(hl_functions) is not int):
            assert not (hl_sizes is not None and hl_functions is not None and len(hl_sizes) < len(hl_functions)), "Sizes must be greater than or equal to functions"
        else:
            assert not (type(hl_sizes) is int and type(hl_functions) is tuple), "Sizes must be greater than or equal to functions"

        #Check each individual size
        if hl_sizes is not None:
            #See if sizes is int (single value), tuple or list. If not, raise
            if type(hl_sizes) is int:
                self.hl_sizes = ((hl_sizes,))
            elif type(hl_sizes) is tuple or type(hl_sizes) is list:
                self.hl_sizes = hl_sizes
            else:
                raise TypeError("Sizes must be either an int (single value), or a list/tuple.")
            
            #See if each size is int and >= 1
            for size in self.hl_sizes:
                try:
                    size = int(size)
                    assert size >= 1, "Hidden layer size must be int greater than 0"
                except ValueError:
                    raise ValueError('Size must be of type int')
                
            #See if functions is str (single value), tuple or list. If not, raise
            if hl_functions is not None:
                if type(hl_functions) is str:
                    hl_functions = ((hl_functions,))
                elif type(hl_functions) is tuple or type(hl_functions) is list:
                    pass
                else:
                    raise TypeError("Functions must be either a str (single value), or a list/tuple.")
                
                #See if each function is str and part of predefined list
                for a_f in hl_functions:
                    assert type(a_f) is str, "Function must be of type str"
                    assert a_f is None or a_f in neural_network.funcs, "Function must be one of the following: " + ", ".join(neural_network.funcs)

        #Assign to instance variables
        self.in_size = in_size
        self.out_size = out_size

        #Set hyperparameters
        self.alpha = alpha
        self.alpha_max = 0.01
        self.alpha_adjust_factor = 10
        
        self.epochs = epochs
        self.threshold = threshold
        self.bias = bias
        self.e_max = 709

        self.loss = 'cross-entropy'
        #Create list of activation functions
        self.a_f = list()
        if hl_functions is not None:
            for a_f in hl_functions:
                self.a_f.append(a_f)
        if out_func is not None:
            self.a_f.append(out_func)
                
        #Create weights
        try:
            self.create_architecture(self.in_size, self.out_size, self.hl_sizes)
        except AttributeError:
            self.create_architecture(self.in_size, self.out_size)


    def train(self, train_input, train_target, test_input, test_target):
        self.train_input = train_input
        self.train_target = train_target
        self.test_input = test_input
        self.test_target = test_target

        for j in range(self.epochs + 1):
            
            # First, feed forward through the hidden layer
            self.feed_forward(self.train_input)
            
            # Then, error back propagation from output to input
            self.back_propagation()

            # Finally, updating the weights of the network
            self.update_weights()

            # From time to time, reporting the results
            if (j % 1) == 0:
                train_error = np.mean(np.abs(self.output_error))
                print('Epoch {:5}'.format(j), end=' - ')
                print('error: {:0.4f}'.format(train_error), end= ' - ')

                train_accuracy = self.accuracy(target=self.train_target, predictions=(self.get_predictions(self.train_target)))
                test_preds = self.predict(self.test_input)
                test_accuracy = self.accuracy(target=self.test_target, predictions=test_preds)

                print('acc: train {:0.3f}'.format(train_accuracy), end= ' | ')
                print('test {:0.3f}'.format(test_accuracy))


    def init_weights(self, arch):
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

        
    def feed_forward(self, inputs):
        #pdb.set_trace()
        #Empty return list
        outputs = {}
        
        #Dot product of input value and weight
        z1 = np.dot(inputs, self.weights['W1'])# + self.weights['b1']
        
        #Input is now equal to activation of output
        a1 = self.activation_func(z1, 'sigmoid')

        #Dot product of hidden layer out and weight
        z2 = np.dot(a1, self.weights['W2'])# + self.weights['b2']

        #Run through softmax
        a2 = self.activation_func(z2, 'softmax')

        
        outs = {"Z1": z1, "A1": a1, "Z2": z2, "A2": a2}
        self.outputs = outs

        
    def activation_func(self, input_values, name='sigmoid'):
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
            input_values = np.where(input_values > self.e_max, self.e_max, input_values)
            input_values = np.where(input_values < -self.e_max, -self.e_max, input_values)
            return 1/(1 + np.exp(-input_values))
        elif name == 'tanh':
            return np.tanh(input_values)
        elif name == 'softmax':
            t = np.exp(input_values)
            a = np.exp(input_values) / np.sum(t, axis=1).reshape(-1,1)
            return a


    def activation_func_prime(self, input_values, name='sigmoid'):
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

            
    
    def back_propagation(self):
        self.deltas = {}
        
        self.output_error = self.calculate_error(self.train_target, self.outputs['A2'])
        error_gradient = self.error_derivative(self.train_target, self.outputs['A2'])
        out_delta = np.dot(self.outputs['A1'].T, error_gradient) / error_gradient.shape[0]
        prior_error = error_gradient
        self.deltas['dW2'] = out_delta
        self.deltas['db2'] = np.sum(error_gradient, axis=0, keepdims=True) / error_gradient.shape[0]

        hidden_out_error = np.dot(error_gradient, self.weights['W2'].T)
        hidden_error = hidden_out_error * self.outputs['A1'] * self.activation_func_prime(self.outputs['A1'])
        hidden_delta = np.matmul(self.train_input.T, hidden_error)
        self.deltas['dW1'] = hidden_delta
        self.deltas['db1'] = np.sum(hidden_error, axis=0, keepdims=True) / error_gradient.shape[0]
        
        
    def calculate_error(self, target, output):
        rows, cols = output.shape
        reshaped_target = np.zeros((rows, 10))
        reshaped_target[np.arange(reshaped_target.shape[0]), target]=1
        #Cost - average loss of each output node
        ce = -np.mean(np.sum(reshaped_target * np.log(output + 1e-8), axis=1))
        return ce

    def error_derivative(self, target, output):
        rows, cols = output.shape
        reshaped_target = np.zeros((rows, 10))
        reshaped_target[np.arange(reshaped_target.shape[0]), target]=1
        return output - reshaped_target

        
    def update_weights(self):
        self.weights['W2'] = self.weights['W2'] - (self.alpha * self.deltas['dW2'])
        self.weights['b2'] = self.weights['b2'] - (self.alpha * self.deltas['db2'])
        self.weights['W1'] = self.weights['W1'] - (self.alpha * self.deltas['dW1'])
        self.weights['b1'] = self.weights['b1'] - (self.alpha * self.deltas['db1'])
        
    def accuracy(self, target, predictions):
        correct_preds = np.sum(predictions.astype(int))
        return correct_preds / len(target)



    def predict(self, inputs):
        self.feed_forward(inputs)
        preds = self.get_predictions(target=self.test_target).astype(int)
        return preds



    def get_predictions(self, target=None):
        predicts = np.argmax(self.outputs['A2'], axis=1)
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

print("--- %s seconds ---" % (time.time() - start_time))
