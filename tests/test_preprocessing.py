from __future__ import unicode_literals

import keras
import pytest

from keras_text.data import Dataset
from keras_text.processing import WordTokenizer, pad_sequences, SentenceWordTokenizer, unicodify


def test_token_preprocessing(tmpdir):
    tokenizer = WordTokenizer()

    X = ['hello', 'world', 'welcome', 'earth']
    y = [0, 1, 0, 1]

    tokenizer.build_vocab(X)

    assert len(tokenizer.token_index) == 4

    X_enc = tokenizer.encode_texts(X)
    X_fin = pad_sequences(X_enc, max_tokens=50)
    y_fin = keras.utils.to_categorical(y, num_classes=2)

    ds = Dataset(X_fin, y_fin, tokenizer=tokenizer)
    ds.update_test_indices(test_size=0.5)

    path = str(tmpdir.mkdir("data").join("test"))

    ds.save(path)

    ds_new = Dataset.load(path)

    # only first word
    assert(all([a == b for a, b in zip(ds_new.X[0], X_fin[0])]))


def test_sentence_tokenizer():
    texts = [
        "HELLO world hello. How are you today? Did you see the S.H.I.E.L.D?",
        "Quick brown fox. Ran over the, building 1234?",
    ]

    texts = unicodify(texts)
    tokenizer = SentenceWordTokenizer()
    tokenizer.build_vocab(texts)
    tokenizer.apply_encoding_options(max_tokens=5)
    encoded = tokenizer.encode_texts(texts)
    decoded = tokenizer.decode_texts(encoded, inplace=False)

    assert(len(decoded) == 2)

    decoded_flat = sum(sum(decoded, []), [])

    assert(len(set(decoded_flat)) == 5)
