import numpy as np
from src.layers .linearlayer import LinearLayer
from src.utils.getActivation import getActivation

class MLP(object):

    def __init__(self, input_dim, output_dim, hidden_dim, hidden_layers = 1, activation_type = 'sigmoid'):
        '''
            Args:
            - activation_type: 'sigmoid', 'tanh'
        '''
        
        # note: do not instantiate activation fn here, if we do this, then all activation fns in all 
        # layers will refer to the same instance.
        self.activation = getActivation(activation_type)
        
        self.modules = [LinearLayer(input_dim, hidden_dim), self.activation()]
        
        for i in range(hidden_layers):
            # for each hidden layer, we also need to add an activation function
            self.modules.append(LinearLayer(hidden_dim, hidden_dim))
            self.modules.append(self.activation())
            
        self.modules.append(LinearLayer(hidden_dim, output_dim)) # last output layer
      

    def predict(self, x):
        # Forward pass:
        for mod in self.modules:
            x = mod.forward(x)
        return x, mod.w
    

    def backprop(self, dy):
        dy = dy.reshape(dy.shape[0],-1)
        for mod in reversed(self.modules):
            dy = mod.backward(dy)
    