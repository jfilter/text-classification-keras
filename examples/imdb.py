import sys

from texcla import corpus, data, experiment
from texcla.models import AlexCNN, AttentionRNN, BasicRNN, StackedRNN, TokenModelFactory, YoonKimCNN
from texcla.preprocessing import FastTextWikiTokenizer

# 1. `python imdb.py setup`: Setup and preprocess the data
# 2. `python imdb.py train`: Load the setup data and train

# truncate text input after 50 tokens (words)
MAX_LEN = 50

def setup():
    # limit to 5k pos. and 5k neg. samples (each for train and test)
    X_train, X_test, y_train, y_test = corpus.imdb(5000)

    # use the special tokenizer used for constructing the embeddings
    tokenizer = FastTextWikiTokenizer()

    # build vocabulary only on training data
    tokenizer = experiment.setup_data(
        X_train, y_train, tokenizer, 'imdb_train.bin', max_len=MAX_LEN)
    experiment.setup_data(X_test, y_test, tokenizer,
                          'imdb_test.bin', max_len=MAX_LEN)


def train():
    ds_train = data.Dataset.load('imdb_train.bin')
    ds_val = data.Dataset.load('imdb_test.bin')

    # use the embedding trained on Simple English Wikipedia
    factory = TokenModelFactory(
        ds_train.num_classes, ds_train.tokenizer.token_index, max_tokens=MAX_LEN, embedding_type='fasttext.wiki.simple', embedding_dims=300)

    word_encoder_model = YoonKimCNN()
    # word_encoder_model = AttentionRNN()
    # word_encoder_model = StackedRNN()
    # word_encoder_model = BasicRNN()

    # freeze word embeddings
    model = factory.build_model(
        token_encoder_model=word_encoder_model, trainable_embeddings=False)

    # use experiment.train as wrapper for Keras.fit()
    experiment.train(x=ds_train.X, y=ds_train.y, validation_data=(ds_val.X, ds_val.y), model=model,
                     word_encoder_model=word_encoder_model)

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    if sys.argv[1] == 'setup':
        setup()
    if sys.argv[1] == 'train':
        train()
