import numpy as np

class Sigmoid(object):
    ''' Sigmoid activation function (output values are between 0 and 1)
    '''

    def __init__(self):
        self.grad = None


    def forward(self, x):
        f = 1.0/(np.exp(-x) + 1)
        self.grad = (1 - f) * f
        return f


    def backward(self, dx):
        ''' dx is downstream gradient, need to compute dx * grad to return to upstream.
            Assume dx and stored grad are of the same dimension.
        '''
        return self.grad * dx