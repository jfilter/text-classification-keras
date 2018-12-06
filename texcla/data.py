from __future__ import absolute_import

import logging

import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

from .utils import io, sampling

logger = logging.getLogger(__name__)


class Dataset(object):

    def __init__(self, X, y, tokenizer=None):
        """Encapsulates all pieces of data to run an experiment. This is basically a bag of items that makes it
        easy to serialize and deserialize everything as a unit.

        Args:
            X: The raw model inputs. This can be set to None if you dont want
                to serialize this value when you save the dataset.
            y: The raw output labels.
            tokenizer: The optional test indices to use. Ideally, this should be generated one time and reused
                across experiments to make results comparable. `generate_test_indices` can be used generate first
                time indices.
            **kwargs: Additional key value items to store.
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.tokenizer = tokenizer

        self.is_multi_label = isinstance(y[0], (set, list, tuple))
        if self.is_multi_label:
            self.label_encoder = MultiLabelBinarizer()
            self.y = self.label_encoder.fit_transform(self.y)
        else:
            self.label_encoder = LabelBinarizer()
            self.label_encoder = self.label_encoder.fit(self.y)
            if (len(self.labels) == 2):
                # https://stackoverflow.com/questions/31947140/sklearn-labelbinarizer-returns-vector-when-there-are-2-classes
                self.y = np.array(
                    [[1, 0] if l == self.labels[0] else [0, 1] for l in self.y])
            else:
                self.y = self.label_encoder.transform(self.y)

    def save(self, file_path):
        """Serializes this dataset to a file.

        Args:
            file_path: The file path to use.
        """
        io.dump(self, file_path)

    @staticmethod
    def load(file_path):
        """Loads the dataset from a file.

        Args:
            file_path: The file path to use.

        Returns:
            The `Dataset` instance.
        """
        return io.load(file_path)

    @property
    def labels(self):
        return self.label_encoder.classes_

    @property
    def num_classes(self):
        if len(self.y.shape) == 1:
            return 1
        else:
            return len(self.labels)
