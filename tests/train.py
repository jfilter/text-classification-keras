import os
import sys

import keras
import numpy as np
import pytest

from texcla.corpus import imdb
from texcla.data import Dataset
from texcla.models import AttentionRNN, StackedRNN, TokenModelFactory, YoonKimCNN
from texcla.preprocessing import SpacyTokenizer
from texcla import experiment

max_len = 50


def test_train():
    X, y, _, _ = imdb(10)

    # use the special tokenizer used for constructing the embeddings
    tokenizer = SpacyTokenizer()

    # preprocess data (once)
    experiment.setup_data(X, y, tokenizer, 'data.bin', max_len=100)

    # load data
    ds = Dataset.load('data.bin')

    # construct base
    factory = TokenModelFactory(
        ds.num_classes, ds.tokenizer.token_index, max_tokens=100,
        embedding_type='glove.6B.50d', embedding_dims=50)

    # choose a model
    word_encoder_model = YoonKimCNN()

    # build a model
    model = factory.build_model(
        token_encoder_model=word_encoder_model, trainable_embeddings=False)

    # use experiment.train as wrapper for Keras.fit()
    experiment.train(x=ds.X, y=ds.y, validation_split=0.1, model=model,
                     word_encoder_model=word_encoder_model, epochs=1, batch_size=32)
