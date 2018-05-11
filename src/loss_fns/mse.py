import numpy as np

def mse(y_pred, y, w, regularization = None, l = 0.1):
    ''' Assume y_pred and y have the same dim and can be batched.
        w: weights
        regularization: None, 'l2'
        l: regularization parameter (lambda)
    '''

    b_size = y_pred.shape[0]
    diff = y_pred.reshape(b_size,-1) - y.reshape(b_size,-1)
    mse = np.power(diff, 2).mean()
    
    dLoss = 2 * diff # dLoss/dy_pred
    
    # computes regularization:
    if regularization == None:
        reg = 0
        dReg = 0
    if regularization == 'l2':
        reg, dReg = ridgeReg(w, l)
    
    
    # combines regularization terms and returns loss & dLoss:
    loss = mse + reg
    dLoss += dReg
    return loss, dLoss


def ridgeReg(w, l = 0.1):
    ''' returns regularization term and its derivative
    '''
    reg = np.power(w, 2).mean() * l
    dReg = 2 * l * w.mean()
    
    return reg, dReg