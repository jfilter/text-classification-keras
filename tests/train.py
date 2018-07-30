import os
import sys

import keras
import numpy as np
import pytest

from texcla.corpus import imdb
from texcla.data import Dataset
from texcla.models import AttentionRNN, StackedRNN, TokenModelFactory, YoonKimCNN
from texcla.preprocessing import SpacyTokenizer

max_len = 50


def test_train():
    X, y, _, _ = imdb(10)

    tokenizer = SpacyTokenizer()

    tokenizer.build_vocab(X)

    # only select top 10k tokens
    tokenizer.apply_encoding_options(limit_top_tokens=20)

    X_encoded = tokenizer.encode_texts(X)
    X_padded = tokenizer.pad_sequences(
        X_encoded, fixed_token_seq_length=max_len)

    y_cat = keras.utils.to_categorical(y, num_classes=2)

    ds = Dataset(X_padded, y_cat, tokenizer=tokenizer)

    X_train, _, y_train, _ = ds.train_val_split()

    # RNN models can use `max_tokens=None` to indicate variable length words per mini-batch.
    factory = TokenModelFactory(
        2, ds.tokenizer.token_index, max_tokens=max_len, embedding_type='glove.6B.300d')
    word_encoder_model = YoonKimCNN(dropout_rate=0)
    model = factory.build_model(
        token_encoder_model=word_encoder_model, trainable_embeddings=False)

    sgd = keras.optimizers.SGD(
        lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.1)
