import numpy as np
from src.loss_fns.mse import mse
from src.models.mlp import MLP

def batch_gd(x_train, y_train, mlp, loss, regularization='l2', l = 0.2, grad_clip=1.5, lr = 1e-4, epochs = 3000, print_every=100):
    ''' 
        Args:
        regularization: None, 'l2'
        l: regularization parameter (lambda)
    '''
    
    for i in range(epochs):
        y_pred, w = mlp.predict(x_train)
        mse, dLoss = loss(y_pred, y_train, w, regularization=regularization, l=l)
        mlp.backprop(dLoss)
        
        # update:
        for mod in mlp.modules:
            if hasattr(mod, 'w'):
                mod.w -= np.clip(mod.dw,-grad_clip, grad_clip)*lr
            if hasattr(mod, 'b'):
                mod.b -= np.clip(mod.db,-grad_clip, grad_clip)*lr            
    
        if i%print_every == 0:
            print('iteration {}, loss:{}'.format(i, mse))