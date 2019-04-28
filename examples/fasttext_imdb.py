import sys

from texcla import corpus, data, experiment
from texcla.models import TokenModelFactory, AveragingEncoder
from texcla.preprocessing import SpacyTokenizer


# FIXME: NOT WORKING. PLEASE FIX ME. There seem to be something wrong with the n-gram features.

# 1. `python imdb.py setup`: Setup and preprocess the data
# 2. `python imdb.py train`: Load the setup data and train

# truncate text input after 50 tokens (words)
MAX_LEN = 400
N_GRAMS = 2
EMB_DIMS = 50
EPOCHS = 5
WORDS_LIMIT = 20000


def setup():
    # limit to 5k pos. and 5k neg. samples (each for train and test)
    X_train, X_test, y_train, y_test = corpus.imdb(1000)

    # use the special tokenizer used for constructing the embeddings
    tokenizer = SpacyTokenizer()

    # build vocabulary only on training data
    tokenizer = experiment.setup_data(
        X_train, y_train, tokenizer, 'imdb_train.bin', max_len=MAX_LEN, ngrams=N_GRAMS, limit_top_tokens=WORDS_LIMIT)
    experiment.setup_data(X_test, y_test, tokenizer,
                          'imdb_test.bin', max_len=MAX_LEN)


def train():
    ds_train = data.Dataset.load('imdb_train.bin')
    ds_val = data.Dataset.load('imdb_test.bin')

    factory = TokenModelFactory(
        ds_train.num_classes, ds_train.tokenizer.token_index, max_tokens=MAX_LEN, embedding_dims=EMB_DIMS, embedding_type=None)

    word_encoder_model = AveragingEncoder()

    # freeze word embeddings
    model = factory.build_model(
        token_encoder_model=word_encoder_model, trainable_embeddings=True)

    # use experiment.train as wrapper for Keras.fit()
    experiment.train(x=ds_train.X, y=ds_train.y, validation_data=(ds_val.X, ds_val.y), model=model,
                     word_encoder_model=word_encoder_model, epochs=EPOCHS)


if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    if sys.argv[1] == 'setup':
        setup()
    if sys.argv[1] == 'train':
        train()
