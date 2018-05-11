class Tanh(object):
    '''Tanh activation function (output values are between -1 and 1)
    '''
    def __init__(self):
        self.grad = None
        
    def forward(self, x):
        f = np.tanh(x)
        self.grad = 1.0/(np.power(x,2) + 1)
        return f
    
    def backward(self, dx):
        ''' dx is downstream gradient, need to compute dx * grad to return to upstream.
            Assume dx and stored grad are of the same dimension.
        '''
        return self.grad * dx