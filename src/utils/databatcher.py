import numpy as np

def data_batcher(x, y, batch_size, shuffle = False):
    ''' Generate batches of training data x and y in x_list and y_list,
        each containing 'batch_size' number of training examples.
    '''
    data_len = x.shape[0]

    # reshape checks:
    x = x.reshape(data_len, -1)
    y = y.reshape(data_len, -1)
    # check shapes of x and y:
    assert x.shape[0] == y.shape[0], 'x and y shapes are not the same.'

    # shuffle data index:
    if shuffle:
        idx = np.arange(data_len)
        np.random.shuffle(idx)
        x = x[idx]
        y = y[idx]

    # get number of batches:
    num_batches, remainder = divmod(data_len, batch_size)
    x_list = np.array_split(x[:data_len - remainder,:], num_batches)
    y_list = np.array_split(y[:data_len - remainder, :], num_batches)

    if remainder != 0:
        x_list.append(x[data_len - remainder:])
        y_list.append(y[data_len - remainder:])
        
    return x_list, y_list