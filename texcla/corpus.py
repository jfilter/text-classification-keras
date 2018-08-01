import os
import io

import keras
import sklearn


def read_folder(directory):
    """read text files in directory and returns them as array

    Args:
        directory: where the text files are

    Returns:
        Array of text
    """
    res = []
    for filename in os.listdir(directory):
        with io.open(os.path.join(directory, filename), encoding="utf-8") as f:
            content = f.read()
            res.append(content)
    return res


def read_pos_neg_data(path, folder, limit):
    """returns array with positive and negative examples"""
    training_pos_path = os.path.join(path, folder, 'pos')
    training_neg_path = os.path.join(path, folder, 'neg')

    X_pos = read_folder(training_pos_path)
    X_neg = read_folder(training_neg_path)

    if limit is None:
        X = X_pos + X_neg
    else:
        X = X_pos[:limit] + X_neg[:limit]

    y = [1] * int(len(X) / 2) + [0] * int(len(X) / 2)

    return X, y


def imdb(limit=None, shuffle=True):
    """Downloads (and caches) IMDB Moview Reviews. 25k training data, 25k test data

    Args:
        limit: get only first N items for each class

    Returns:
        [X_train, y_train, X_test, y_test]
    """

    movie_review_url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

    # download and extract, thus remove the suffix '.tar.gz'
    path = keras.utils.get_file(
        'aclImdb.tar.gz', movie_review_url, extract=True)[:-7]

    X_train, y_train = read_pos_neg_data(path, 'train', limit)
    X_test, y_test = read_pos_neg_data(path, 'test', limit)

    if shuffle:
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        X_test, y_test = sklearn.utils.shuffle(X_test, y_test)

    return X_train, X_test, y_train, y_test
