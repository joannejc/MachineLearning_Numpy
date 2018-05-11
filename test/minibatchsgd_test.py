import numpy as np
from context import src
from src.models.mlp import MLP
from src.optimizers.miniBatchsgd import mini_batch_sgd
from src.loss_fns.mse import mse

def main():
    # test data:
    x = np.random.randn(100,5)
    # example: y = x1*x2*x3^3 + x4/x5 - x5
    y = np.asarray(x[:,0]*x[:,1] * np.power(x[:,2],3) + x[:,3]/x[:,4] - x[:,4])

    mlp = MLP(5, 1, 30, 2, activation_type= 'tanh')
    mini_batch_sgd(x, y, 64, mlp, mse,regularization=None, l = 0.2, shuffle = True, lr = 8e-3, print_every = 200, epochs= 5000)

    #print(vars(mlp))

if __name__ == "__main__":
    main()