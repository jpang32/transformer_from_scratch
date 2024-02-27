import numpy as np


class Layer:

    def forward(self, inputs: np.array):
        raise NotImplementedError()
