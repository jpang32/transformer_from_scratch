import json


if __name__ == "__main__":
    """ Script to clean the vocab json used for training gpt2 (https://huggingface.co/openai-community/gpt2/tree/main)
    
    The original vocab contains many odd unicode characters that make it difficult to do a look up for certain words
    
    The most common of these is \u0120, which is the only character that this script removes for now. This script may be
    altered in the future to remove more of these characters (see https://github.com/openai/gpt-2/issues/80).
    """

    with open("gpt2_vocab.json") as input_file:
        vocab = json.load(input_file)

    cleaned_vocab_words = set([
        word.replace("\u0120", "")
        for word in vocab
    ])

    out = dict(zip(cleaned_vocab_words, list(range(len(cleaned_vocab_words)))))

    with open("vocab.json", "w") as out_file:
        json.dump(out, out_file)