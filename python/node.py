import numpy as np

class node:
    """
    Node class

    Instance variables
    ------------------
    Name : (self.name)
        str, syntax is first letter of layer + #
    Layer : (self.layer)
        str, I, O, H#
    Activation function : (self.a_f)
        str, name of the activation function
    List of weights associated : (self.weights)
        np.array, array of associated weight objects
        Created after all nodes created for efficiency.
    """

    id_count = 0
    
    def __init__(self, layer, name, a_f=None):
        assert layer is not None, "Layer must be a str, syntax: first letter of layer (I, H#, O)"
        assert name is not None, "Name must be a str, syntax: first letter of layer, # (I1, H12, O3)"
        assert type(layer) is str, "Layer must be a string"
        assert type(name) is str, "Name must be a string"
        assert type(a_f) is str or a_f is None, "Activation function (a_f) must be a str"

        
        self.a_f = a_f if a_f is not None else 'Sigmoid'

        
#Testing purposes
if __name__ == '__main__':
    n = node('a', 'b', "a")
    n1 = node('b', 'c')
    
