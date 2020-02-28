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
                #self.alpha = self.alpha * self.alpha_adjust_factor if self.alpha < self.alpha_max else self.alpha_max
                print('Alpha is {}'.format(self.alpha))
                train_error = np.mean(np.abs(self.output_error))
                print('Epoch {:5}'.format(j), end=' - ')
                print('error: {:0.4f}'.format(train_error), end= ' - ')

                train_accuracy = self.accuracy(target=self.train_target, predictions=(self.get_predictions(self.train_target)))
                test_preds = self.predict(self.test_input)
                test_accuracy = self.accuracy(target=self.test_target, predictions=test_preds)

                print('acc: train {:0.3f}'.format(train_accuracy), end= ' | ')
                print('test {:0.3f}'.format(test_accuracy))

            print('Epoch {} done'.format(j))
    def init_weights(self, inp, out):
        #randn creates random element, divide by squareroot of inp for randomness
        if self.bias:
            return np.random.randn(inp+1, out) / np.sqrt(inp)
        else:
            return np.random.randn(inp, out) / np.sqrt(inp)
    
    
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
        
        #Create a random seed
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
        self.weights = [self.init_weights(inp, out) for inp, out in arch]
            

        
    def feed_forward(self, inputs):
        #Create copy of test data
        i_c = inputs.copy()
            
        #Empty return list
        out = list()
        
        #get activation function range
        a_f_range = len(self.weights) - len(self.a_f)

        
        for W in range(len(self.weights)):
            #If bias exists, add an input for each bias
            
            if self.bias:
                rows, _ = inputs.shape
                ones = np.ones((rows, 1))
                i_c = np.hstack((i_c, ones))
            
            #Dot product of input value and weight
            #pdb.set_trace()
            z = np.dot(i_c, self.weights[W])

            #Check if there is an activation function for this layer
            if len(self.a_f) > 0 and W >= a_f_range and self.a_f[W-len(self.a_f)]:
                #Input is now equal to activation of output
                i_c = self.activation_func(z, self.a_f[W-len(self.a_f)])
            else:
                i_c = z
                
            #Append new input to return
            out.append(i_c)
            
        self.outputs = out


        
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
        """
        self.deltas = list()
        #pdb.set_trace()
        
        if self.bias:
            
            self.bias_outputs = list()
            self.bias_deltas = list()
            self.bias_weights = list()
            
            for weight in range(len(self.weights)):
                self.bias_weights.append(self.weights[weight][-1])
                self.weights[weight] = np.delete(self.weights[weight], -1, 0)
                
        #Get error or cost for this set
        pdb.set_trace()
        self.output_error = self.calculate_error(self.train_target, self.outputs[-1])

        #Cost times derivative is gradient
        output_delta = self.output_error * self.activation_func_prime(self.outputs[-1])
        
        #For every available output down, need to get the error, and delta.
        #Start at 2nd to last output (last hidden layer), and work backwords
        prior_delta = output_delta
        self.deltas.append(output_delta)

        
        if self.bias:
            #Need to make inputs of shape
            rows, cols = self.outputs[-1].shape
            ones = np.ones((rows, cols))
            bias_outputs = ones * self.bias_weights[-1]
            bias_layer_delta = self.output_error * self.activation_func_prime(bias_outputs)
            self.bias_outputs.append(bias_outputs)
            self.bias_deltas.append(bias_layer_delta)
             
        for layer in range(len(self.outputs) - 2, -1, -1):                
            layer_error = prior_delta.dot(self.weights[layer + 1].T)
            layer_delta = layer_error * self.activation_func_prime(self.outputs[layer])

            
            #Backpropagate to bias if included
            if self.bias:
                #Need to make inputs of shape
                rows, cols = self.outputs[layer].shape
                ones = np.ones((rows, cols))
                bias_outputs = ones * self.bias_weights[layer]
                bias_layer_delta = layer_error * self.activation_func_prime(bias_outputs)
                self.bias_outputs.append(bias_outputs)
                self.bias_deltas.append(bias_layer_delta)
             
            prior_delta = layer_delta
            self.deltas.append(layer_delta)
            
        #Put them in order
        self.deltas.reverse()
        if self.bias:
            self.bias_deltas.reverse()
            self.bias_outputs.reverse()
        """
        self.deltas = list()
        self.output_error = self.calculate_error(self.train_target, self.outputs[-1])
        error_gradient = self.error_derivative(self.train_target, self.outputs[-1])
        #np.dot(error_gradient[0].reshape(-1,1), self.outputs[0][0].reshape(1, 500))
        #rows = error_gradient.shape[0]
        #cols = self.outputs[0][0].shape[0]
       # out_delta = np.array([np.dot(error_gradient[row].reshape(-1,1), self.outputs[0][row].reshape(1, cols)) for row in range(rows)])
        #out_delta = np.mean(out_delta, axis=0)
        #wi,j = yi * (output(j) - target(j))
        #weights from node 1 in hidden layer to output node 10:
        #self.outputs[0](col 1) * (mean(error_gradient[9]))
        #self.outputs[0][i] * np.mean(self.outputs[1][:,j] * error_gradient[:,j])
        #np.multiply(self.outputs[0][:,0], error_gradient[:,0])
        deltas = np.dot(self.outputs[0].T, error_gradient) / error_gradient.shape[0]
        #pdb.set_trace()
        #self.activation_func_prime(self.outputs[-1], self.a_f[-1])
        prior_deltas = deltas
        self.deltas.append(prior_deltas)

        for layer in range(len(self.outputs)-2, -1, -1):
            layer_errors = prior_deltas * self.weights[layer+1]
            errors_summed = np.sum(layer_errors, axis=1)
            if layer != 0:
                layer_deltas = np.dot(self.activation_func_prime(self.outputs[layer], self.a_f[layer]),
                                      errors_summed.reshape(-1, 1)) * self.outputs[layer-1]
            else:
               #pdb.set_trace()
                layer_gradients = np.multiply(self.activation_func_prime(self.outputs[layer], self.a_f[layer]),
                                      errors_summed)
                layer_deltas = np.dot(self.train_input.T, layer_gradients)
            self.deltas.append(layer_deltas)
            prior_deltas = layer_deltas
        self.deltas.reverse()

        
    def calculate_error(self, target, output):
        #Target is a #, output is an array of probabilities (softmax)
        if self.loss == 'mse':
            rows, cols = output.shape
            reshaped_target = np.zeros((rows, 10))
            reshaped_target[np.arange(reshaped_target.shape[0]), target]=1
            
            return (np.square(reshaped_target - output)).mean(axis=1)
        elif self.loss == 'cross-entropy':
            rows, cols = output.shape
            reshaped_target = np.zeros((rows, 10))
            reshaped_target[np.arange(reshaped_target.shape[0]), target]=1
            #Cost - average loss of each output node
            ce = -np.sum(reshaped_target * np.log(output + 1e-8), axis=1) / cols
            return ce

    def error_derivative(self, target, output):
        if self.loss == 'cross-entropy':
            rows, cols = output.shape
            reshaped_target = np.zeros((rows, 10))
            reshaped_target[np.arange(reshaped_target.shape[0]), target]=1
            return output - reshaped_target

        
    def update_weights(self):
        #pdb.set_trace()
        for layer in range(len(self.weights)-1, -1, -1):
            #self.weights[layer] = self.weights[layer] + (self.alpha * self.outputs[layer-1].T.dot(self.deltas[layer]))
            self.weights[layer] = self.weights[layer] + (self.alpha * self.deltas[layer])
            if self.bias:
                    self.bias_weights[layer] = self.bias_weights[layer] + (self.alpha * self.bias_outputs[layer].T.dot(self.bias_deltas[layer]))

        #pdb.set_trace()            
        #self.weights[0] = self.weights[0] + (self.alpha * self.train_input.T.dot(self.deltas[0]))
        #self.weights[0] = self.weights[0] + (self.alpha * self.deltas[0])
        if self.bias:
            self.bias_weights[0] = self.bias_weights[0] + (self.alpha * self.bias_outputs[0].T.dot(self.bias_deltas[0]))

        if self.bias:
            for layer in range(len(self.weights)):
                self.bias_weights[layer] = np.mean(self.bias_weights[layer], axis=0)
                self.weights[layer] = np.vstack((self.weights[layer], self.bias_weights[layer]))


    def accuracy(self, target, predictions):
        #Get the correct predictions, then compare to target.)
        correct_preds = np.ravel(predictions)==target
        return np.sum(correct_preds) / len(target)



    def predict(self, inputs):
        #pdb.set_trace()
        
        self.feed_forward(inputs)
        preds = self.get_predictions(target=self.test_target).astype(int)
        return preds



    def get_predictions(self, target=None):
        #pdb.set_trace()
        if len(self.a_f) == 0 or self.a_f[-1] == 'sigmoid':
            return self.outputs[-1] > self.threshold
        elif self.a_f[-1] == 'softmax':
            #return np.ravel(self.outputs[-1])==target
            ret_tup = []
            for output in range(len(self.outputs[-1])):
                guess_right = np.argmax(self.outputs[-1][output])==target[output]
                ret_tup.append((guess_right))
            return np.array(ret_tup)

np.random.seed(0)
rootDir = Path(sys.path[0]).parent
train_images = str(rootDir) + "\\MNIST test data\\train-images.idx3-ubyte"
train_label = str(rootDir) + "\\MNIST test data\\train-labels.idx1-ubyte"
test_images = str(rootDir) + "\\MNIST test data\\t10k-images.idx3-ubyte"
test_label = str(rootDir) + "\\MNIST test data\\t10k-labels.idx1-ubyte"

train_image_data = idx2numpy.convert_from_file(train_images)
train_label_data = idx2numpy.convert_from_file(train_label)
test_image_data = idx2numpy.convert_from_file(test_images)
test_label_data = idx2numpy.convert_from_file(test_label)

items, rows, cols = train_image_data.shape
train_image_data = train_image_data.reshape(items, rows * cols)
items, rows, cols = test_image_data.shape
test_image_data = test_image_data.reshape(items, rows * cols)
_, in_nodes = train_image_data.shape

nn = neural_network(in_nodes, 10,
                    out_func='softmax',
                    hl_sizes=500,
                    hl_functions='sigmoid',
                    alpha=0.05,
                    epochs=5, bias=False)

start_time = time.time()
nn.train(train_input=train_image_data,
         train_target=train_label_data,
         test_input=test_image_data,
         test_target=test_label_data)

print("--- %s seconds ---" % (time.time() - start_time))
