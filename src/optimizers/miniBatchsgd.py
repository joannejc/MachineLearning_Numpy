import numpy as np
from src.loss_fns.mse import mse
from src.models.mlp import MLP
from src.utils .databatcher import data_batcher

def mini_batch_sgd(x_train, y_train, batch_size, mlp, loss, grad_clip=1.5, lr = 1e-4, epochs = 3000, print_every=100, shuffle=True):
    ''' mini-batch sgd
    '''
        
    for i in range(epochs):
        x, y = data_batcher(x_train, y_train, batch_size, shuffle) # x, y are lists of arrays
        #lr = lr/len(x_train)
        for k, xk in enumerate(x):
            yk_pred = mlp.predict(xk)
            mse, dLoss = loss(yk_pred, y[k])
            mlp.backprop(dLoss)
            
            # update:
            for mod in mlp.modules:
                if hasattr(mod, 'w'):
                    mod.w -= np.clip(mod.dw,-grad_clip, grad_clip)*lr
                if hasattr(mod, 'b'):
                    mod.b -= np.clip(mod.db,-grad_clip, grad_clip)*lr
                
        
        if i%print_every == 0:
            print('iteration {}, loss:{}'.format(i, mse))