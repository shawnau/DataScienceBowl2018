import numpy as np


# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))
