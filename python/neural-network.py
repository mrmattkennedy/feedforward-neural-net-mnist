import math
import numpy as np
from node import node
from weight import weight

class neural_network:
    """
    Neural Network class

    Instance variables
    ------------------
    nodes : (self.nodes)
        np.array, matrix of all nodes
    weights : (self.weights)
        np.array, matrix of all weights
    Learning rate : (self.alpha)
        float, learning rate

    Constructor
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


    
    def __init__(self, in_size, out_size, out_func=None, hl_count=None, **hl_s_af):
        #Verify sizes are numbers above 0
        assert type(in_size) is int, "Size of input layer needs to be an int"
        assert type(out_size) is int, "Size of output layer needs to be an int"
        assert in_size >= 1, "Size of input layer must be a positive integer"
        assert out_size >= 1, "Size of output layer must be a positive integer"
        assert out_func is None or out_func in neural_network.funcs, "Function must be one of the following: " + ", ".join(neural_network.funcs)
        
        #Verify hidden layer elements
        assert len(hl_s_af) == 2, "Hidden layer vars must have 2 items: Sizes and activation functions"
        hl_vals = [value for value in hl_s_af.values()]

        #Check each size
        for size in hl_vals[0]:
            try:
                size = int(size)
                assert size >= 1, "Hidden layer size must be int greater than 0"
            except ValueError:
                raise ValueError('Size must be of type int')
        #Check each function
        for a_f in hl_vals[1]:
            assert a_f is None or a_f in neural_network.funcs, "Function must be one of the following: " + ", ".join(neural_network.funcs)

        
    
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
            
hidden_layer_vals = {"Sizes" : ("5", "10"), "Functions" : ("softmax", "tanh")}
nn = neural_network(1, 1, **hidden_layer_vals)
#node1 = node("a", "b")
#node2 = node("a", "b")
#weight1 = weight(node1, node2, 1.0)
#arr = 200 * np.random.random_sample((10, 1)) - 100
#nn = neural_network()
#arr = nn.activation_function(arr, name='Tanh')
#print(arr)
