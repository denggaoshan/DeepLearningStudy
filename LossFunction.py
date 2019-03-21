import numpy as np

def MSELoss(x, y):
    assert x.shape == y.shape
    return np.linalg.norm( x - y) ** 2

#TODO
def CrossEntropyLoss(x, y):
    assert x.shape == y.shape
    pass