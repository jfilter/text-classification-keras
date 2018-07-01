# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import logging
import os
import io

import numpy as np
import six
from keras.utils.data_utils import get_file

logger = logging.getLogger(__name__)
_EMBEDDINGS_CACHE = dict()

# Add more types here as needed.
# – fastText: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
# - glove: https://nlp.stanford.edu/projects/glove/

_EMBEDDING_TYPES = {
    # simple English
    'fasttext.simple': {
        'file': 'fasttext.simple.vec',
        'url': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
    },
    # English
    'fasttext.en': {
        'file': 'fasttext.en.vec',
        'url': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec'
    },
    # 42 Billion tokens Common Crawl
    'glove.42B.300d': {
        'file': 'glove.42B.300d.txt',
        'url': 'http://nlp.stanford.edu/data/glove.42B.300d.zip'
    },
    # 6 Billion tokens from Wikipedia 2014 + Gigaword 5
    'glove.6B.50d': {
        'file': 'glove.6B.50d.txt',
        'url': 'http://nlp.stanford.edu/data/glove.6B.zip'
    },

    'glove.6B.100d': {
        'file': 'glove.6B.100d.txt',
        'url': 'http://nlp.stanford.edu/data/glove.6B.zip'
    },

    'glove.6B.200d': {
        'file': 'glove.6B.200d.txt',
        'url': 'http://nlp.stanford.edu/data/glove.6B.zip'
    },

    'glove.6B.300d': {
        'file': 'glove.6B.300d.txt',
        'url': 'http://nlp.stanford.edu/data/glove.6B.zip'
    },
    #  840 Billion tokens Common Crawl
    'glove.840B.300d': {
        'file': 'glove.840B.300d.txt',
        'url': 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
    },
    # 2 Billion tweets, 27 Billion tokens Twitter
    'glove.twitter.27B.25d': {
        'file': 'glove.twitter.27B.25d.txt',
        'url': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
    },
    'glove.twitter.27B.50d': {
        'file': 'glove.twitter.27B.50d.txt',
        'url': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
    },
    'glove.twitter.27B.100d': {
        'file': 'glove.twitter.27B.100d.txt',
        'url': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
    },
    'glove.twitter.27B.200d': {
        'file': 'glove.twitter.27B.200d.txt',
        'url': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
    },
}


def _build_embeddings_index(embeddings_path, embedding_dims):
    logger.info('Building embeddings index...')
    index = {}
    with io.open(embeddings_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()

            # some hack for fasttext vectors where the first line is (num_token, dimensions)
            if len(values) <= 2 and embedding_dims > 1:
                continue

            word = ' '.join(values[:-embedding_dims])
            floats = values[-embedding_dims:]

            if not isinstance(word, six.text_type):
                word = word.decode()

            vector = np.asarray(floats, dtype='float32')
            index[word] = vector
    logger.info('Done')
    return index


def build_embedding_weights(word_index, embeddings_index):
    """Builds an embedding matrix for all words in vocab using embeddings_index
    """
    logger.info('Loading embeddings for all words in the corpus')
    embedding_dim = list(embeddings_index.values())[0].shape[-1]

    # setting special tokens such as UNK and PAD to 0
    # all other words are also set to 0.
    embedding_weights = np.zeros((len(word_index), embedding_dim))

    for word, i in word_index.items():
        word_vector = embeddings_index.get(word)
        if word_vector is not None:
            embedding_weights[i] = word_vector

    return embedding_weights


def build_fasttest_embedding_obj(embedding_type):
    """All fasttext vectors
    Args:
        embedding_type: A string in the format `fastext.$LANG_CODE`. e.g. `fasttext.de` or `fasttest.es`
    Returns:
        name and url
    """
    lang = embedding_type.split('.')[1]
    return {
        'file': 'wiki.{}.vec'.format(lang),
        'url': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.{}.vec'.format(lang),
        'extract': False,
    }


def get_embedding_type(embedding_type):
    if embedding_type.startswith('fasttext.'):
        return build_fasttest_embedding_obj(embedding_type)

    data_obj = _EMBEDDING_TYPES.get(embedding_type)

    if data_obj is None:
        raise ValueError("Embedding name should be one of '{}'".format(
            _EMBEDDING_TYPES.keys()))

    return data_obj


def get_embeddings_index(embedding_type='glove.42B.300d', embedding_path=None, embedding_dims=None):
    """Retrieves embeddings index from embedding name or path. Will automatically download and cache as needed.

    Args:
        embedding_type: The embedding type to load.
        embedding_path: Path to a local embedding to use instead of the embedding type. Ignores `embedding_type` if specified.

    Returns:
        The embeddings indexed by word.
    """

    if embedding_path is not None:
        embedding_type = embedding_path  # identify embedding by path

    embeddings_index = _EMBEDDINGS_CACHE.get(embedding_type)
    if embeddings_index is not None:
        return embeddings_index

    if embedding_path is None:
        embedding_type_obj = get_embedding_type(embedding_type)
        file_path = get_file(
            embedding_type_obj['file'], origin=embedding_type_obj['url'], extract=embedding_type_obj.get('extract', True), cache_subdir='embeddings')
    else:
        file_path = embedding_path

    embeddings_index = _build_embeddings_index(file_path, embedding_dims)
    _EMBEDDINGS_CACHE[embedding_type] = embeddings_index
    return embeddings_index
