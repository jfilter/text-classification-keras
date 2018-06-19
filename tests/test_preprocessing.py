import keras
import pytest

from keras_text.data import Dataset
from keras_text.processing import WordTokenizer, pad_sequences


def test_pre(tmpdir):
    tokenizer = WordTokenizer()

    X = ['hello', 'world']
    y = [0, 1]

    tokenizer.build_vocab(X)

    assert len(tokenizer.token_index) == 2

    X_enc = tokenizer.encode_texts(X)
    X_fin = pad_sequences(X_enc, max_tokens=50)
    y_fin = keras.utils.to_categorical(y, num_classes=2)

    ds = Dataset(X_fin, y_fin, tokenizer=tokenizer)
    ds.update_test_indices(test_size=0.1)

    path = tmpdir.mkdir("data").join("test")

    ds.save(path)

    ds_new = Dataset.load(path)

    assert ds_new.X == X
