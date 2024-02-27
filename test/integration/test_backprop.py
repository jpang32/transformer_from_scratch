import numpy as np
import pytest
from src.linear_layer import LinearLayer
from src.softmax import Softmax

def test_backprop():
    # Add linear layer with shape to process vocab of input
    # layer1 = LinearLayer(shape=())
    # layer2 = Sigmoid(LinearLayer(shape=()))
    # pass values through sigmoid layer
    # Calculate loss w.r.t to sigmoid output
    # Propagate loss backwards through layers
    # Assert that weights have been changed
    # Ideally, also assert that the output that you get over a couple of
    # iterations creates a correlation between certain words before and after other
    # words (e.g., "the" before nouns in a sentence)

    input_sentence = "This is a test sentence for a test function in a test suite"
    input_sentence_list = input_sentence.lower().split()

    vocab = set(input_sentence_list)
    vocab_indices = dict(zip(vocab, range(len(vocab))))

    window_size = 3
    hidden_dimensions = 3
    layer1 = LinearLayer(shape=(len(vocab), hidden_dimensions))
    layer2 = Softmax(LinearLayer(shape=(hidden_dimensions, len(vocab) * 2 * window_size)))
    for word in input_sentence_list:
        one_hot_encoded_word = np.zeros(len(vocab), dtype=int)
        one_hot_encoded_word[vocab_indices[word]] = 1
        out = layer1.forward(one_hot_encoded_word)
        out = layer2.forward(out)
        print(out)
        # After this, calculate loss
