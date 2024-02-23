from typing import Union

import numpy as np


class Differentiable:

    def __init__(self):
        self._accumulated_gradient = 0

    def derivative(self, parameters: Union[int, float, np.array]):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()