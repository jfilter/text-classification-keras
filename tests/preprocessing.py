from __future__ import unicode_literals

import keras
import pytest

from keras_text.data import Dataset
from keras_text.preprocessing import SpacyTokenizer, SpacySentenceTokenizer, TwokenizeTokenizer, SimpleTokenizer
from keras_text.preprocessing.utils import unicodify


def test_token_preprocessing(tmpdir):
    tokenizer = SpacyTokenizer()

    X = ['hello', 'world', 'welcome', 'earth']
    y = [0, 1, 0, 1]

    tokenizer.build_vocab(X)

    assert(len(tokenizer.token_index) - len(tokenizer.special_token) == 4)

    X_enc = tokenizer.encode_texts(X)
    X_fin = tokenizer.pad_sequences(X_enc, fixed_token_seq_length=50)
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
    unicodify(texts)
    tokenizer = SpacySentenceTokenizer()
    tokenizer.build_vocab(texts)
    tokenizer.apply_encoding_options(limit_top_tokens=5)
    encoded = tokenizer.encode_texts(texts)
    decoded = tokenizer.decode_texts(encoded, inplace=False)

    assert(len(decoded) == 2)

    decoded_flat = sum(sum(decoded, []), [])

    assert(len(set(decoded_flat)) == 6)  # 5 + 1 for <UNK>


def test_padding():
    texts = [
        "HELLO world hello.",
        "Quick brown fox. Ran over the, building 1234?",
        "Peter is a cool guy.",
    ]

    texts = unicodify(texts)
    tokenizer = SpacyTokenizer()
    tokenizer.build_vocab(texts[:-1])

    encoded = tokenizer.encode_texts(texts)
    padded = tokenizer.pad_sequences(encoded, fixed_token_seq_length=7)

    assert(len(padded[0]) == 7)
    assert(len(padded[1]) == 7)
    assert(len(padded[2]) == 7)

    decoded = tokenizer.decode_texts(padded, inplace=False)
    print(decoded)
    assert('guy' not in decoded[-1])


def test_twokenizer():
    texts = [
        "HELLO world hello.",
        "Quick brown fox. Ran over the, building 1234 1.2.3.5?",
        "Peter is a cool guy.",
    ]
    tokenizer = TwokenizeTokenizer()
    tokenizer.build_vocab(texts)
    assert('1.2.3.5' in tokenizer.token_index)
    assert('1' not in tokenizer.token_index)


def test_simple_tokenizer():
    texts = [
        "HELLO world hello.",
        "Quick brown fox. Ran over the, building 1234 1.2.3.5?",
        "Peter is a cool guy.",
    ]
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts)
    assert('fox.' in tokenizer.token_index)
    assert(' ' not in tokenizer.token_index)
