import os
import sys

import keras
import numpy as np
import pytest

from keras_text.corpus import imdb
from keras_text.data import Dataset
from keras_text.models import AttentionRNN, StackedRNN, TokenModelFactory, YoonKimCNN, AlexCNN
from keras_text.processing import WordTokenizer

max_len = 50

path = 'imdb_proc_data.bin'


def build_dataset():
    X, y, _, _ = imdb(1000)

    tokenizer = WordTokenizer()

    tokenizer.build_vocab(X)

    # only select top 5k tokens
    tokenizer.apply_encoding_options(limit_top_tokens=5000)

    X_encoded = tokenizer.encode_texts(X)
    X_padded = tokenizer.pad_sequences(
        X_encoded, fixed_token_seq_length=max_len)

    y_cat = keras.utils.to_categorical(y, num_classes=2)

    print(y_cat[:10])

    ds = Dataset(X_padded, y_cat, tokenizer=tokenizer)
    ds.update_test_indices(test_size=0.1)
    ds.save(path)


def train():
    ds = Dataset.load(path)
    X_train, _, y_train, _ = ds.train_val_split()

    print(ds.tokenizer.decode_texts(X_train[:10]))

    print(y_train[:10])

    # RNN models can use `max_tokens=None` to indicate variable length words per mini-batch.
    factory = TokenModelFactory(
        2, ds.tokenizer.token_index, max_tokens=max_len, embedding_type='glove.6B.300d')

    word_encoder_model = YoonKimCNN()
    # word_encoder_model = AlexCNN(dropout_rate=[0, 0])
    # word_encoder_model = AttentionRNN()
    # word_encoder_model = StackedRNN()
    model = factory.build_model(
        token_encoder_model=word_encoder_model, trainable_embeddings=False)


    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)


def main():
    if len(sys.argv) != 2:
        raise 'error'
    if sys.argv[1] == 'build':
        build_dataset()
    if sys.argv[1] == 'train':
        train()


if __name__ == '__main__':
    main()
