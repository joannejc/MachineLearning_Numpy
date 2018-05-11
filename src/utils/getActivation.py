from src.activation_fns.sigmoid import Sigmoid
from src.activation_fns.tanh import Tanh

def getActivation(fnName):
    '''
        Arg: fnName is the activation function name
        e.g. sigmoid, tanh
    '''
    if fnName == 'sigmoid':
        return Sigmoid
    if fnName == 'tanh':
        return Tanh
    else:
        raise ValueError('invalid activation fn name.')