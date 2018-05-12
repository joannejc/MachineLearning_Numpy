import numpy as np

"""
class GetSGDStepSize(object):
    '''
    '''
    def __init__(step, k, lr_type):
        '''
            Args:
            step: step size/ learning rate at k-th iteration
            k: iteration number
            lr_type: step size update type (constant, decay)
        '''
        self.step = step
        self.k = k
        self.lr_type = lr_type
    
    
    def 
"""

def getSGDStepSize(step, i, lr_type):
    '''
        Args:
        step: step size/ learning rate at k-th iteration
        i: iteration number
        lr_type: step size update type ('constant', 'decay')
    '''
    if lr_type == 'constant':
        step = step
    if lr_type == 'decay':
        if i == 0:
            step = step
        else:
            step = 1 - 1.0/i 
    return step
    
