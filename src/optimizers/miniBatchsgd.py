import numpy as np
from src.loss_fns.mse import mse
from src.models.mlp import MLP
from src.utils .databatcher import data_batcher
from src.utils.getSGDStepSize import getSGDStepSize

def mini_batch_sgd(x_train, y_train, batch_size, mlp, loss, regularization = None, l= 0.1, lr_type = 'constant', grad_clip=1.5, lr = 0.1, epochs = 3000, print_every=100, shuffle=True):
    ''' 
        Args:
        regularization: None, 'l2'
        l: regularization parameter (lambda)
        lr_type = 'constant' or 'decay' step size
    '''
    step = lr
        
    for i in range(epochs):
        x, y = data_batcher(x_train, y_train, batch_size, shuffle) # x, y are lists of arrays
        step = getSGDStepSize(step, i, lr_type)

        for k, xk in enumerate(x):
            yk_pred, wk = mlp.predict(xk)
            mse, dLoss = loss(yk_pred, y[k], wk, regularization = regularization, l=l)
            mlp.backprop(dLoss)
            
            # update:
            for mod in mlp.modules:
                if hasattr(mod, 'w'):
                    mod.w -= np.clip(mod.dw,-grad_clip, grad_clip)*lr
                if hasattr(mod, 'b'):
                    mod.b -= np.clip(mod.db,-grad_clip, grad_clip)*lr
                
        
        if i%print_every == 0:
            print('iteration {}, loss:{}'.format(i, mse))