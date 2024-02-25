from typing import Union

import numpy as np


class Differentiable:

    def __init__(self, train_mode=False):
        self._accumulated_gradient = 0
        self.train_mode = train_mode

    def derivative(self, parameters: Union[int, float, np.array]):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()