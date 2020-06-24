import numpy as np

def get_train_val_test(data, train_index, val_index, test_index):
    return data[train_index], data[val_index], data[test_index]

def sigmoid(x):
    return 1/(1+np.exp(-x))
