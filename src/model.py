import numpy as np

from layer import Layer


class Model:

    def __init__(self):
        self._layers = []

    def add_layer(self, layer: Layer):
        self._layers.append(layer)

    @property
    def layers(self):
        return self._layers

    def forward(self, input: np.array):
        out = input
        for layer in self.layers:
            out = layer(out)

    # def train(self, epochs=10, learning_rate=0.01):
    # 
    # def step(self):
    # 
    # def calculate_gradients(self):
        