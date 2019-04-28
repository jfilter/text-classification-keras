from __future__ import absolute_import, unicode_literals

import abc
import logging
from collections import OrderedDict, defaultdict
from copy import deepcopy
from multiprocessing import cpu_count

import numpy as np
import six
from keras.preprocessing.sequence import pad_sequences as keras_pad_sequences
from keras.utils.generic_utils import Progbar

from ..libs import ngrams

from . import utils
from ..utils import io

try:
    import spacy
except ImportError:
    pass


logger = logging.getLogger(__name__)


class Tokenizer(object):

    def __init__(self,
                 lang='en',
                 lower=True,
                 special_token=['<PAD>', '<UNK>']):  # 0 - Pad, 1 - Unkown
        """Encodes text into `(samples, aux_indices..., token)` where each token is mapped to a unique index starting
        from `i`. `i` is the number of special tokens.

        Args:
            lang: The spacy language to use. (Default value: 'en')
            lower: Lower cases the tokens if True. (Default value: True)
            special_token: The tokens that are reserved. Default: ['<UNK>', '<PAD>'], <UNK> for unknown words and <PAD> for padding token.

        """

        self.lang = lang
        self.lower = lower
        self.special_token = special_token

        self._token2idx = dict()
        self._idx2token = dict()
        self._token_counts = defaultdict(int)

        self._num_texts = 0
        self._counts = None

    @abc.abstractmethod
    def token_generator(self, texts, **kwargs):
        """Generator for yielding tokens. You need to implement this method.

        Args:
            texts: list of text items to tokenize.
            **kwargs: The kwargs propagated from `build_vocab_and_encode` or `encode_texts` call.

        Returns:
            `(text_idx, aux_indices..., token)` where aux_indices are optional. For example, if you want to vectorize
                `texts` as `(text_idx, sentences, words), you should return `(text_idx, sentence_idx, word_token)`.
                Similarly, you can include paragraph, page level information etc., if needed.
        """
        raise NotImplementedError()

    def create_token_indices(self, tokens):
        """If `apply_encoding_options` is inadequate, one can retrieve tokens from `self.token_counts`, filter with
        a desired strategy and regenerate `token_index` using this method. The token index is subsequently used
        when `encode_texts` or `decode_texts` methods are called.
        """
        start_index = len(self.special_token)
        indices = list(range(len(tokens) + start_index))
        # prepend because the special tokens come in the beginning
        tokens_with_special = self.special_token + list(tokens)
        self._token2idx = dict(list(zip(tokens_with_special, indices)))
        self._idx2token = dict(list(zip(indices, tokens_with_special)))

    def apply_encoding_options(self, min_token_count=1, limit_top_tokens=None):
        """Applies the given settings for subsequent calls to `encode_texts` and `decode_texts`. This allows you to
        play with different settings without having to re-run tokenization on the entire corpus.

        Args:
            min_token_count: The minimum token count (frequency) in order to include during encoding. All tokens
                below this frequency will be encoded to `0` which corresponds to unknown token. (Default value = 1)
            limit_top_tokens: The maximum number of tokens to keep, based their frequency. Only the most common `limit_top_tokens`
                tokens will be kept. Set to None to keep everything. (Default value: None)
        """
        if not self.has_vocab:
            raise ValueError("You need to build the vocabulary using `build_vocab` "
                             "before using `apply_encoding_options`")
        if min_token_count < 1:
            raise ValueError("`min_token_count` should atleast be 1")

        # Remove tokens with freq < min_token_count
        token_counts = list(self._token_counts.items())
        token_counts = [x for x in token_counts if x[1] >= min_token_count]

        # Clip to max_tokens.
        if limit_top_tokens is not None:
            token_counts.sort(key=lambda x: x[1], reverse=True)
            filtered_tokens = list(zip(*token_counts))[0]
            filtered_tokens = filtered_tokens[:limit_top_tokens]
        else:
            filtered_tokens = zip(*token_counts)[0]

        # Generate indices based on filtered tokens.
        self.create_token_indices(filtered_tokens)

    def encode_texts(self, texts, unknown_token="<UNK>", verbose=1, **kwargs):
        """Encodes the given texts using internal vocabulary with optionally applied encoding options. See
        ``apply_encoding_options` to set various options.

        Args:
            texts: The list of text items to encode.
            unknown_token: The token to replace words that out of vocabulary. If none, those words are omitted.
            verbose: The verbosity level for progress. Can be 0, 1, 2. (Default value = 1)
            **kwargs: The kwargs for `token_generator`.

        Returns:
            The encoded texts.
        """
        if not self.has_vocab:
            raise ValueError(
                "You need to build the vocabulary using `build_vocab` before using `encode_texts`")

        if unknown_token and unknown_token not in self.special_token:
            raise ValueError(
                "Your special token (" + unknown_token + ") to replace unknown words is not in the list of special token: " + self.special_token)

        progbar = Progbar(len(texts), verbose=verbose, interval=0.25)
        encoded_texts = []
        for token_data in self.token_generator(texts, **kwargs):
            indices, token = token_data[:-1], token_data[-1]

            token_idx = self._token2idx.get(token)
            if token_idx is None and unknown_token:
                token_idx = self.special_token.index(unknown_token)

            if token_idx is not None:
                utils._append(encoded_texts, indices, token_idx)

            # Update progressbar per document level.
            progbar.update(indices[0])

        # All done. Finalize progressbar.
        progbar.update(len(texts))
        return encoded_texts

    def decode_texts(self, encoded_texts, unknown_token="<UNK>", inplace=True):
        """Decodes the texts using internal vocabulary. The list structure is maintained.

        Args:
            encoded_texts: The list of texts to decode.
            unknown_token: The placeholder value for unknown token. (Default value: "<UNK>")
            inplace: True to make changes inplace. (Default value: True)

        Returns:
            The decoded texts.
        """
        if len(self._token2idx) == 0:
            raise ValueError(
                "You need to build vocabulary using `build_vocab` before using `decode_texts`")

        if not isinstance(encoded_texts, list):
            # assume it's a numpy array
            encoded_texts = encoded_texts.tolist()

        if not inplace:
            encoded_texts = deepcopy(encoded_texts)
        utils._recursive_apply(encoded_texts,
                               lambda token_id: self._idx2token.get(token_id) or unknown_token)
        return encoded_texts

    def add_tokens(self, ngram_set):
        start_index = len(self._token2idx) + 1
        print('start: ', start_index)
        tmp = {}
        for k, v in enumerate(ngram_set):
            # print(k, v)
            idx = k + start_index
            self._token2idx[v] = idx
            self._idx2token[idx] = v

            tmp[v] = idx
            # TODO: Counts?
        return tmp

    def add_ngrams(self, encoded_texts, train=False, n=2):
        if train:
            ngram_set = set()
            for input_list in encoded_texts:
                for i in range(2, n + 1):
                    set_of_ngram = ngrams.create_ngram_set(
                        input_list, ngram_value=i)
                    ngram_set.update(set_of_ngram)
            print(list(ngram_set)[:1000])
            ngram_set = [x for x in ngram_set if 1 not in x]
            print(ngram_set[:1000])
            tmp = self.add_tokens(ngram_set)

        return ngrams.add_ngram(encoded_texts, token_indice=tmp, ngram_range=n)

    def build_vocab(self, texts, verbose=1, **kwargs):
        """Builds the internal vocabulary and computes various statistics.

        Args:
            texts: The list of text items to encode.
            verbose: The verbosity level for progress. Can be 0, 1, 2. (Default value = 1)
            **kwargs: The kwargs for `token_generator`.
        """
        if self.has_vocab:
            logger.warn(
                "Tokenizer already has existing vocabulary. Overriding and building new vocabulary.")

        progbar = Progbar(len(texts), verbose=verbose, interval=0.25)
        count_tracker = utils._CountTracker()

        self._token_counts.clear()
        self._num_texts = len(texts)

        for token_data in self.token_generator(texts, **kwargs):
            indices, token = token_data[:-1], token_data[-1]
            count_tracker.update(indices)
            self._token_counts[token] += 1

            # Update progressbar per document level.
            progbar.update(indices[0])

        # Generate token2idx and idx2token.
        self.create_token_indices(self._token_counts.keys())

        # All done. Finalize progressbar update and count tracker.
        count_tracker.finalize()
        self._counts = count_tracker.counts
        progbar.update(len(texts))

    def pad_sequences(self, sequences, fixed_sentences_seq_length=None, fixed_token_seq_length=None,
                      padding='pre', truncating='post', padding_token="<PAD>"):
        """Pads each sequence to the same fixed length (length of the longest sequence or provided override).

        Args:
            sequences: list of list (samples, words) or list of list of list (samples, sentences, words)
            fixed_sentences_seq_length: The fix sentence sequence length to use. If None, largest sentence length is used.
            fixed_token_seq_length: The fix token sequence length to use. If None, largest word length is used.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger than fixed_sentences_seq_length or fixed_token_seq_length
                either in the beginning or in the end of the sentence or word sequence respectively.
            padding_token: The token to add for padding.

        Returns:
            Numpy array of (samples, max_sentences, max_tokens) or (samples, max_tokens) depending on the sequence input.

        Raises:
            ValueError: in case of invalid values for `truncating` or `padding`.
        """
        value = self.special_token.index(padding_token)
        if value < 0:
            raise ValueError('The padding token "' + padding_token +
                             " is not in the special tokens of the tokenizer.")
        # Determine if input is (samples, max_sentences, max_tokens) or not.
        if isinstance(sequences[0][0], list):
            x = utils._pad_sent_sequences(sequences, fixed_sentences_seq_length,
                                          fixed_token_seq_length, padding, truncating, value)
        else:
            x = utils._pad_token_sequences(
                sequences, fixed_token_seq_length, padding, truncating, value)
        return np.array(x, dtype='int32')

    def get_counts(self, i):
        """Numpy array of count values for aux_indices. For example, if `token_generator` generates
        `(text_idx, sentence_idx, word)`, then `get_counts(0)` returns the numpy array of sentence lengths across
        texts. Similarly, `get_counts(1)` will return the numpy array of token lengths across sentences.

        This is useful to plot histogram or eyeball the distributions. For getting standard statistics, you can use
        `get_stats` method.
        """
        if not self.has_vocab:
            raise ValueError(
                "You need to build the vocabulary using `build_vocab` before using `get_counts`")
        return self._counts[i]

    def get_stats(self, i):
        """Gets the standard statistics for aux_index `i`. For example, if `token_generator` generates
        `(text_idx, sentence_idx, word)`, then `get_stats(0)` will return various statistics about sentence lengths
        across texts. Similarly, `get_counts(1)` will return statistics of token lengths across sentences.

        This information can be used to pad or truncate inputs.
        """
        # OrderedDict to always show same order if printed.
        result = OrderedDict()
        result['min'] = np.min(self._counts[i])
        result['max'] = np.max(self._counts[i])
        result['std'] = np.std(self._counts[i])
        result['mean'] = np.mean(self._counts[i])
        return result

    def save(self, file_path):
        """Serializes this tokenizer to a file.

        Args:
            file_path: The file path to use.
        """
        io.dump(self, file_path)

    @staticmethod
    def load(file_path):
        """Loads the Tokenizer from a file.

        Args:
            file_path: The file path to use.

        Returns:
            The `Dataset` instance.
        """
        return io.load(file_path)

    @property
    def has_vocab(self):
        return len(self._token_counts) > 0 and self._counts is not None

    @property
    def token_index(self):
        """Dictionary of token -> idx mappings. This can change with calls to `apply_encoding_options`.
        """
        return self._token2idx

    @property
    def token_counts(self):
        """Dictionary of token -> count values for the text corpus used to `build_vocab`.
        """
        return self._token_counts

    @property
    def num_tokens(self):
        """Number of unique tokens for use in enccoding/decoding.
        This can change with calls to `apply_encoding_options`.
        """
        return len(self._token2idx)

    @property
    def num_texts(self):
        """The number of texts used to build the vocabulary.
        """
        return self._num_texts
