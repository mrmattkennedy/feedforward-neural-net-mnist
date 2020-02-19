import math
import numpy as np

class neural_network:
    def __init__(self):
        return
    
    def activation_function(self, input_values, name='Sigmoid'):
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

        if name == 'Sigmoid':
            return np.array([1 / (1 + math.exp(-x)) for x in input_values])
        elif name == 'Tanh':
            return np.tanh(input_values)
            

arr = 200 * np.random.random_sample((10, 1)) - 100

nn = neural_network()
arr = nn.activation_function(arr, name='Tanh')
print(arr)
