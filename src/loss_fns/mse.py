import numpy as np

def mse(y_pred, y):
    ''' Assume y_pred and y have the same dim and can be batched.
    '''
    b_size = y_pred.shape[0]
    diff = y_pred.reshape(b_size,-1) - y.reshape(b_size,-1)
    mse = np.power(diff, 2).mean()
    
    dLoss = 2 * diff # dLoss/dy_pred
    
    return mse, dLoss