import numpy as np


# Ideally, would like to
# 1) process tokens from data
# 2) Initiate model using tokens, vocab, etc.
# 3) Write model.train()

class SkipGramEmbeddingModel:

    def __init__(self, tokens: list[str], vocab: list[str], hidden_dim: int = 512, window_size=3):
        # For each token, locate its index in the vocab
        # Initialize weight matrix with len(vocab) rows, hidden_dim cols
        # Initialize weight matrix with hidden_dim cols, window_size * 2 * len(vocab) cols

        # For each token in tokens, do a forward pass, then calculate loss
        # Do backprop only on those outputs that correspond to an actual part of the window
        self.tokens = tokens
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.window_size = window_size

        self.hidden = np.random.rand(len(vocab), hidden_dim)
        self.output = np.random.rand(hidden_dim, 2 * window_size * len(vocab))

    def _loss(self):
        NotImplemented()

    def train(self, epochs=10):
        # Calculate output
        # Use output to calculate loss
        # use loss to calculate backprop needed

        for i, token in enumerate(self.tokens):
            left_window_index = i - self.window_size if i - self.window_size >= 0 else 0
            right_window_index = i + self.window_size if i + self.window_size < len(self.tokens) else len(self.tokens) - 1
            embedding = self.hidden[self.vocab.index(token)]

            # Calculate the output for each future / past word
            # Do backpropagation, only focusing on those weights which were used to calculate output
                # In this case, does it make sense to use arrays of weight() values as opposed to numpy arrays?
