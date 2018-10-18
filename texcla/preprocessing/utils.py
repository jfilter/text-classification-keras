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

try:
    import spacy
except ImportError:
    pass


logger = logging.getLogger(__name__)


class _CountTracker(object):
    """Helper class to track counts of various document hierarchies in the corpus.
    For example, if the tokenizer can tokenize docs as (docs, paragraph, sentences, words), then this utility
    will track number of paragraphs, number of sentences within paragraphs and number of words within sentence.
    """

    def __init__(self):
        self._prev_indices = None
        self._local_counts = None
        self.counts = None

    def update(self, indices):
        """Updates counts based on indices. The algorithm tracks the index change at i and
        update global counts for all indices beyond i with local counts tracked so far.
        """
        # Initialize various lists for the first time based on length of indices.
        if self._prev_indices is None:
            self._prev_indices = indices

            # +1 to track token counts in the last index.
            self._local_counts = np.full(len(indices) + 1, 1)
            self._local_counts[-1] = 0
            self.counts = [[] for _ in range(len(self._local_counts))]

        has_reset = False
        for i in range(len(indices)):
            # index value changed. Push all local values beyond i to count and reset those local_counts.
            # For example, if document index changed, push counts on sentences and tokens and reset their local_counts
            # to indicate that we are tracking those for new document. We need to do this at all document hierarchies.
            if indices[i] > self._prev_indices[i]:
                self._local_counts[i] += 1
                has_reset = True
                for j in range(i + 1, len(self.counts)):
                    self.counts[j].append(self._local_counts[j])
                    self._local_counts[j] = 1

        # If none of the aux indices changed, update token count.
        if not has_reset:
            self._local_counts[-1] += 1
        self._prev_indices = indices[:]

    def finalize(self):
        """This will add the very last document to counts. We also get rid of counts[0] since that
        represents document level which doesnt come under anything else. We also convert all count
        values to numpy arrays so that stats can be computed easily.
        """
        for i in range(1, len(self._local_counts)):
            self.counts[i].append(self._local_counts[i])
        self.counts.pop(0)

        for i in range(len(self.counts)):
            self.counts[i] = np.array(self.counts[i])


def _apply_generator(texts, apply_fn):
    for text in texts:
        yield apply_fn(text)


def _append(lst, indices, value):
    """Adds `value` to `lst` list indexed by `indices`. Will create sub lists as required.
    """
    for i, idx in enumerate(indices):
        # We need to loop because sometimes indices can increment by more than 1 due to missing tokens.
        # Example: Sentence with no words after filtering words.
        while len(lst) <= idx:
            # Update max counts whenever a new sublist is created.
            # There is no need to worry about indices beyond `i` since they will end up creating new lists as well.
            lst.append([])
        lst = lst[idx]

    # Add token and update token max count.
    lst.append(value)


def _recursive_apply(lst, apply_fn):
    if len(lst) > 0 and not isinstance(lst[0], list):
        for i in range(len(lst)):
            lst[i] = apply_fn(lst[i])
    else:
        for sub_list in lst:
            _recursive_apply(sub_list, apply_fn)


def _to_unicode(text):
    if not isinstance(text, six.text_type):
        text = text.decode('utf-8')
    return text


def _parse_spacy_kwargs(**kwargs):
    """Supported args include:

    Args:
        n_threads/num_threads: Number of threads to use. Uses num_cpus - 1 by default.
        batch_size: The number of texts to accumulate into a common working set before processing.
            (Default value: 1000)
    """
    n_threads = kwargs.get('n_threads') or kwargs.get('num_threads')
    batch_size = kwargs.get('batch_size')

    if n_threads is None or n_threads is -1:
        n_threads = cpu_count() - 1
    if batch_size is None or batch_size is -1:
        batch_size = 1000
    return n_threads, batch_size


def _pad_token_sequences(sequences, max_tokens,
                         padding, truncating, value):
    # TODO: better variable names (see below)
    return keras_pad_sequences(sequences, maxlen=max_tokens, padding=padding, truncating=truncating, value=value)


def _pad_sent_sequences(sequences, max_sentences, max_tokens, padding, truncating, value):
    # TODO: better names (see below)
    # Infer max lengths if needed.
    if max_sentences is None or max_tokens is None:
        max_sentences_computed = 0
        max_tokens_computed = 0
        for sent_seq in sequences:
            max_sentences_computed = max(max_sentences_computed, len(sent_seq))
            max_tokens_computed = max(max_tokens_computed, np.max(
                [len(token_seq) for token_seq in sent_seq]))

        # Only use inferred values for None.
        if max_sentences is None:
            max_sentences = max_sentences_computed

        if max_tokens is None:
            max_tokens = max_tokens_computed

    result = np.ones(shape=(len(sequences), max_sentences, max_tokens)) * value

    for idx, sent_seq in enumerate(sequences):
        # empty list/array was found
        if not len(sent_seq):
            continue
        if truncating == 'pre':
            trunc = sent_seq[-max_sentences:]
        elif truncating == 'post':
            trunc = sent_seq[:max_sentences]
        else:
            raise ValueError(
                'Truncating type "%s" not understood' % truncating)

        # Apply padding.
        if padding == 'post':
            result[idx, :len(trunc)] = _pad_token_sequences(
                trunc, max_tokens, padding, truncating, value)
        elif padding == 'pre':
            result[idx, -len(trunc):] = _pad_token_sequences(trunc,
                                                             max_tokens, padding, truncating, value)
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return result


def unicodify(texts):
    """Encodes all text sequences as unicode. This is a python2 hassle.

    Args:
        texts: The sequence of texts.

    Returns:
        Unicode encoded sequences.
    """
    return [_to_unicode(text) for text in texts]
