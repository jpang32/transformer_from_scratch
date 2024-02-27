from typing import Union

import numpy as np

from src.differentiable import Differentiable
from src.layer import Layer


def _softmax(inputs: np.array):
    """ Applies softmax function to a vector of weights

    :param inputs: vector of weights
    :return: vector of softmax values
    """
    assert inputs.ndim == 1

    denom = np.sum(np.exp(inputs))
    return np.exp(inputs) / denom


class Softmax(Differentiable, Layer):

    def __init__(self, layer: Layer):
        super().__init__()
        self.layer = layer

    def forward(self, inputs: np.array) -> np.array:
        """ Applies softmax to the outputs of the layer

        :param inputs: vector of weights
        :return: vector of softmax values
        """
        out = _softmax(self.layer.forward(inputs))
        if self.train_mode:
            self.derivatives = self.derivative(out)

        return _softmax(self.layer.forward(inputs))

    def derivative(self, out: np.array) -> np.array:
        return -np.outer(out, out)
