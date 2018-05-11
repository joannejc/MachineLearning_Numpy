import numpy as np

class LinearLayer(object):
    ''' affine transform on input x with parameters w and b: y = wx + b
        input_dim is number of variables (dim of x)
    '''
    def __init__(self, input_dim, output_dim, weight_init = 'randn'):
        ### add util to have all diff weight_init methods
        ### assume below wt initializing method is what we have for now:
        self.w = np.random.randn(input_dim, output_dim) * 0.001
        self.b = np.random.randn(1, output_dim) * 0
        self.x = None
        
        self.dw = None
        self.db = None

    
    def forward(self, x):
        self.x = x
        y = np.matmul(x, self.w) + self.b  # computes y = wx + b
        return y
    
    
    def backward(self, dy):
        ''' dy is downstream gradient, need to compute dw, db, and  
            return dx to upstream.
        '''
        self.dw = np.matmul(self.x.T,dy)
        self.db = np.sum(dy, axis = 0, keepdims= True)
        dx = np.matmul(dy, self.w.T)
        return dx
