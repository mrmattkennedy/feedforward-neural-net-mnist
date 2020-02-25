import math
import numpy as np
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
    Activation functions (self.a_f)
        list(str), list of strings of activation functions, with output last
    Constructor parameters
    -----------
    input nodes : in_size
        int, number of input nodes
    output nodes : out_size
        int, number of output nodes
    output function : out_func
        str, name of the output function (default is softmax)
    hidden layers : hl_count
        int, number of hidden layers (can be None)
    Hidden layer vars : **hl_s_af
        strs, activation function for each hidden layer (ignored if h_l_count is None)

    
    Global variables
    ----------------
    Function keywords : funcs
        List(str), allowed function names
    """
    funcs = ['sigmoid', 'tanh', 'softmax', '']


    
    def __init__(self, in_size, out_size,
                 train_input, train_target, test_input, test_target,
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

        #Get test sets
        self.train_input = train_input
        self.train_target = train_target
        self.test_input = test_input
        self.test_target = test_target
        
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


    #def train(self):
        
    def feed_forward(self, inputs):
        #Create copy of test data
        a = inputs.copy()
        #Empty return list
        out = list()
        for W in range(len(self.weights)):
            #Dot product of input value and weight
            z = np.dot(a, self.weights[W])

            #Check if there is an activation function for this layer
            if len(self.a_f) > 0 and W - len(self.a_f) >= 0 and self.a_f[W-len(self.a_f)]:
                #Input is now equal to activation of output
                a = activation_function(self, z, self.a_f[W-len(self.a_f)])
            else:
                a = z
            #Append new input to return
            out.append(a)

        self.ouputs = out


    
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
            return 1/(1 + np.exp(input_values))
        elif name == 'tanh':
            return np.tanh(input_values)



    def activation_func_prime(self, input_values, name='sigmoid'):
        if name == 'sigmoid':
            return input_values * (1 - input_values)
    
    def back_propagation(self):
        #Reshape y
        output_error = self.train_target.reshape(-1, 1) - self.outputs[len(self.outputs)-1]
        output_delta - output_error * activation_func_prime(self.outputs[len(self.outputs)-1])
        l2_delta = l2_error * sigmoid_prime(l2) #Cost times derivative is gradient
        l1_error = l2_delta.dot(weights[1].T)
        l1_delta = l1_error * sigmoid_prime(l1)
        return l2_error, l1_delta, l2_delta


    
coord, cl = make_moons(300, noise=0.05)
X, Xt, y, yt = train_test_split(coord, cl,
                                test_size=0.30,
                                random_state=0)

nn = neural_network(2, 5, hl_sizes=2, train_input=X, train_target=y, test_input=Xt, test_target=yt)
output = nn.feed_forward(X)
print(output)
#nn.back_propogation(outputs


#node1 = node("a", "b")
#node2 = node("a", "b")
#weight1 = weight(node1, node2, 1.0)
#arr = 200 * np.random.random_sample((10, 1)) - 100
#nn = neural_network()
#arr = nn.activation_function(arr, name='Tanh')
#print(arr)
