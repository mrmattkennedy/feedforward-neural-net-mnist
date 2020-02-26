import pdb
import sys
import math
import traceback
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("error")


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
                 alpha=0.05, epochs=30000, threshold=0.5,
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
        self.epochs = epochs
        self.threshold = threshold
        
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

            #for output in self.outputs:
             #s   print(output)
            # Then, error back propagation from output to input
            self.back_propagation()

            # Finally, updating the weights of the network
            self.update_weights()

            # From time to time, reporting the results
            if (j % 5000) == 0:
                train_error = np.mean(np.abs(self.output_error))
                print('Epoch {:5}'.format(j), end=' - ')
                print('error: {:0.4f}'.format(train_error), end= ' - ')

                train_accuracy = self.accuracy(target=self.train_target, predictions=(self.outputs[-1] > self.threshold))
                test_preds = self.predict(self.test_input)
                test_accuracy = self.accuracy(target=self.test_target, predictions=test_preds)

                print('acc: train {:0.3f}'.format(train_accuracy), end= ' | ')
                print('test {:0.3f}'.format(test_accuracy))


        print(self.outputs[-1])           
    def init_weights(self, inp, out):
        #randn creates random element, divide by squareroot of inp for randomness
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
        a = inputs.copy()

        #Empty return list
        out = list()
        
        #get activation function range
        a_f_range = len(self.weights) - len(self.a_f)

        for W in range(len(self.weights)):
            #Dot product of input value and weight
            z = np.dot(a, self.weights[W])

            #Check if there is an activation function for this layer
            if len(self.a_f) > 0 and W >= a_f_range and self.a_f[W-len(self.a_f)]:
                #Input is now equal to activation of output
                a = self.activation_func(z, self.a_f[W-len(self.a_f)])
            else:
                a = z
            
            #Append new input to return
            out.append(a)

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
            return 1/(1 + np.exp(-input_values))
        elif name == 'tanh':
            return np.tanh(input_values)



    def activation_func_prime(self, input_values, name='sigmoid'):
        if name == 'sigmoid':
            return input_values * (1 - input_values)


    
    def back_propagation(self):
        self.deltas = list()
        #Get error or cost for this set
        self.output_error = self.train_target.reshape(-1, 1) - self.outputs[-1]

        #Cost times derivative is gradient
        output_delta = self.output_error * self.activation_func_prime(self.outputs[len(self.outputs)-1])
        
        #For every available output down, need to get the error, and delta.
        #Start at 2nd to last output (last hidden layer), and work backwords
        prior_delta = output_delta
        self.deltas.append(output_delta)

        try:
            for layer in range(len(self.outputs) - 2, -1, -1):
                layer_error = prior_delta.dot(self.weights[layer + 1].T)
                layer_delta = layer_error * self.activation_func_prime(self.outputs[layer])
                #pdb.set_trace()
                prior_delta = layer_delta
                self.deltas.append(layer_delta)
        except RuntimeWarning:
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        #Put them in order
        self.deltas.reverse()


        
    def update_weights(self):
        for layer in range(len(self.weights)-1, 0, -1):
            self.weights[layer] = self.weights[layer] + (self.alpha * self.outputs[layer-1].T.dot(self.deltas[layer]))
            
        self.weights[0] = self.weights[0] + (self.alpha * self.train_input.T.dot(self.deltas[0]))



    def accuracy(self, target, predictions):
        #Get the correct predictions, then compare to target.)
        correct_preds = np.ravel(predictions)==target
        return np.sum(correct_preds) / len(target)



    def predict(self, inputs):
        self.feed_forward(inputs)
        preds = np.ravel(((self.outputs[-1]>self.threshold).astype(int)))
        return preds




np.random.seed(0)
coord, cl = make_moons(300, noise=0.05)
X, Xt, y, yt = train_test_split(coord, cl,
                                test_size=0.30,
                                random_state=0)
print(X)
nn = neural_network(2, 1, out_func='sigmoid', hl_sizes=(3, 4), hl_functions=('sigmoid', 'sigmoid'), epochs=30000)
nn.train(train_input=X, train_target=y, test_input=Xt, test_target=yt)
