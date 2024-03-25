import json

import numpy as np
import pytest


def softmax(vector: np.array):
    return np.exp(vector) / np.sum(np.exp(vector))


def main():
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

    with open("../vocab.json") as vocab_file:
        vocab = json.load(vocab_file)

    reverse_vocab = {v: k for k, v in vocab.items()}

    len_vocab = len(vocab)

    window_size = 3
    hidden_dimensions = 300

    hidden_layer = np.random.rand(len(vocab), hidden_dimensions)
    output_layers = [np.random.rand(hidden_dimensions, len_vocab) for i in range(window_size * 2)]

    # Load vocabulary
    print(output_layers)

    # Translate vocab index into one-hot-encoded vector
    # Run vector through layers to get output
    for word in input_sentence_list:
        word_embedding = hidden_layer[vocab[word], :]
        out = [softmax(word_embedding.dot(layer)) for layer in output_layers]

        out_words = [reverse_vocab[np.argmax(out_vec)] for out_vec in out]

    print(out_words)


if __name__ == "__main__":
    main()
