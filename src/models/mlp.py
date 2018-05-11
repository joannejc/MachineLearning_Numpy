import numpy as np
from src.activation_fns.sigmoid import Sigmoid
from src.layers .linearlayer import LinearLayer

class MLP(object):
    '''
    '''
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_layers = 1, activation_type = 'sigmoid'):
        
        if activation_type == 'sigmoid':
            # note: do not instantiate Sigmoid, if we do this, then all activation fns in all 
            # layers will refer to the same instance.
            self.activtion = Sigmoid ### assume we only have sigmoid for now
        
        self.modules = [LinearLayer(input_dim, hidden_dim), self.activtion()]
        
        for i in range(hidden_layers):
            # for each hidden layer, we also need to add an activation function
            self.modules.append(LinearLayer(hidden_dim, hidden_dim))
            self.modules.append(self.activtion())
            
        self.modules.append(LinearLayer(hidden_dim, output_dim)) # last output layer
      
    def predict(self, x):
        # Forward pass:
        for mod in self.modules:
            x = mod.forward(x)
        return x
    
    def backprop(self, dy):
        dy = dy.reshape(dy.shape[0],-1)
        for mod in reversed(self.modules):
            dy = mod.backward(dy)