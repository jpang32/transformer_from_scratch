import numpy as np

from src.layer import Layer


class LinearLayer(Layer):

    def __init__(self, shape: tuple):
        super().__init__()

        self.shape = shape
        self.weights = np.random.rand(*self.shape)

