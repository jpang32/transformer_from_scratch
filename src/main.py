import json

import numpy as np
import pytest


def softmax(vector: np.array):
    return np.exp(vector) / np.sum(np.exp(vector))


def get_negative_sampling(corpus: str, vocab: dict, num_words: int = 10):
    """ Selects negative samples using distribution defined in negative sampling paper

    This function will not consider words that are not in our vocab.

    For now, the function only splits the corpus on whitespaces, but in the future should
    tokenize the text in order to remove extraneous characters.

    This only needs to be calculated once per corpus - the distribution can be re-used for
    negative sample selection

    :param corpus: corpus of text, represented as a list of strings
    :return: random negative samples
    """
    words_list = corpus.split(" ")

    word_freqs = {}
    for word in words_list:
        if word in vocab:
            if word in word_freqs:
                word_freqs[word] += 1
            else:
                word_freqs[word] = 1

    word_freq_list = [word_freq ** (3 / 4) for word_freq in word_freqs.values()]
    word_freq_sum = sum(word_freq_list)
    distribution = [word_freq / word_freq_sum for word_freq in word_freq_list]

    return np.random.choice(list(word_freqs.keys()), size=num_words, p=distribution).tolist()


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

    with open("../vocab.json", "r") as vocab_file:
        vocab = json.load(vocab_file)

    with open("../data/wiki_movie_plots.txt", "r") as text_file:
        corpus = text_file.read()

    negative_samples = get_negative_sampling(corpus, vocab)
    print(negative_samples)

    reverse_vocab = {v: k for k, v in vocab.items()}

    len_vocab = len(vocab)

    window_size = 3
    hidden_dimensions = 300

    hidden_layer = np.random.rand(len(vocab), hidden_dimensions)
    output_layers = [np.random.rand(hidden_dimensions, len_vocab) for i in range(window_size * 2)]

    # Load vocabulary
    # print(output_layers)

    # Translate vocab index into one-hot-encoded vector
    # Run vector through layers to get output
    for word in input_sentence_list:
        word_embedding = hidden_layer[vocab[word], :]
        out = [softmax(word_embedding.dot(layer)) for layer in output_layers]

        out_words = [reverse_vocab[np.argmax(out_vec)] for out_vec in out]

    print(out_words)


if __name__ == "__main__":
    main()
