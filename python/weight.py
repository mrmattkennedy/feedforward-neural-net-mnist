import node
import numpy as np

class weight:
    """
    Weight class

    Instance variables
    ------------------
    From-node : (self.f_n)
        node, starting node
    To-node : (self.t_n)
        node, ending node
    Weight value : (self.val)
        float, weight value
    """
    
    def __init__(self, f_n, t_n, val):
        assert type(f_n) is node.node, '"From-node" (f_n) must be of type node.node'
        assert type(t_n) is node.node, '"To-node" (t_n) must be of type node.node'
        assert type(val) is float, '"Weight value" (val) must be of type float'

        assert val >= -1.0 and val <= 1.0, '"Weight value" (val) must be between -1.0 and 1.0'
        
        self.f_n = f_n
        self.t_n = t_n
        self.val = val
        
#Testing purposes
#if __name__ == '__main__':
    #n = weight('a', 'b')
    #n1 = node('b', 'c')
    #print(n.name)
    #print(n1.name)
