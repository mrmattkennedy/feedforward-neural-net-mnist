import math
import numpy as np

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
    funcs = ['sigmoid', 'tanh', 'softmax']


    
    def __init__(self, in_size, out_size, out_func=None, hl_sizes=None, hl_functions=None):
        #Verify sizes are numbers above 0
        assert type(in_size) is int, "Size of input layer needs to be an int"
        assert type(out_size) is int, "Size of output layer needs to be an int"
        assert in_size >= 1, "Size of input layer must be a positive integer"
        assert out_size >= 1, "Size of output layer must be a positive integer"
        assert out_func is None or out_func in neural_network.funcs, "Function must be one of the following: " + ", ".join(neural_network.funcs)
        
        #Verify hidden layer elements
        #assert (hl_sizes is None and hl_functions is None) or (hl_sizes is not None and hl_functions is not None), "Must have both sizes and functions for hidden layers"
        assert not (hl_sizes is None and hl_functions is not None), "Must have sizes with functions"
        
        #assert len(hl_sizes) == len(hl_functions)
        if hl_sizes is not None:
            assert len(hl_sizes) == 2, "Hidden layer vars must have 2 items: Sizes and activation functions"
            self.hl_vals = [value for value in hl_s_af.values()]
            assert len(self.hl_vals) == 2, "Hidden layer vars must have 2 items: Sizes and activation functions"
            assert len(self.hl_vals[0]) == len(self.hl_vals[1]), "Length of each item for hidden layer vars must be equal"

            
            #Check each size
            for size in self.hl_vals[0]:
                try:
                    size = int(size)
                    assert size >= 1, "Hidden layer size must be int greater than 0"
                except ValueError:
                    raise ValueError('Size must be of type int')
                
            #Check each function
            for a_f in self.hl_vals[1]:
                assert a_f is None or a_f in neural_network.funcs, "Function must be one of the following: " + ", ".join(neural_network.funcs)

        #Assign to instance variables
        self.in_size = in_size
        self.out_size = out_size

        #Create weights
        try:
            self.create_architecture(self.in_size, self.out_size, self.hl_vals[0])
        except AttributeError:
            self.create_architecture(self.in_size, self.out_size)

        
    def create_architecture(self, in_layer=None, out_layer=None, hidden_layers=None, random_seed=0):
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
        self.weights = [self.init(inp, out) for inp, out in arch]
        print(self.weights)

    
    def init(self, inp, out):
        #randn creates random element, divide by squareroot of inp for randomness
        return np.random.randn(inp, out) / np.sqrt(inp)


    
    def activation_function(self, input_values, name='sigmoid'):
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
            return np.array([1 / (1 + math.exp(-x)) for x in input_values])
        elif name == 'tanh':
            return np.tanh(input_values)
        
nn = neural_network(2, 3)
#node1 = node("a", "b")
#node2 = node("a", "b")
#weight1 = weight(node1, node2, 1.0)
#arr = 200 * np.random.random_sample((10, 1)) - 100
#nn = neural_network()
#arr = nn.activation_function(arr, name='Tanh')
#print(arr)
