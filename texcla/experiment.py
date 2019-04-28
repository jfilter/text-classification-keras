import csv
import datetime
import inspect
import os
import pathlib
from os import path
from shutil import copyfile, move

import deep_plots
import keras
import six
from sklearn.model_selection import train_test_split

from .data import Dataset
from .utils.format import to_fixed_digits


def create_experiment_folder(base_dir, model, lr, batch_size):
    if six.PY2:
        try:
            os.makedirs(base_dir)
        except:
            pass
    else:
        pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)

    num_folders = len(next(os.walk(base_dir))[1])

    # 4 digits
    exp_id = "%05d" % num_folders

    filename = [exp_id, str(model), "lr",
                to_fixed_digits(lr), "bs", to_fixed_digits(batch_size)]
    filename = '_'.join(filename)
    filename = filename.replace('.', '_')

    exp_path = path.join(base_dir, filename)
    pathlib.Path(exp_path).mkdir(parents=True)
    return exp_path


def copy_called_file(exp_path):
    # because it's called within train
    _, filename, _, _, _, _ = inspect.stack()[2]
    copyfile(filename, path.join(
        exp_path, filename.split('/')[-1]))  # only last


def create_callbacks(exp_path, patience):
    checkpoint = keras.callbacks.ModelCheckpoint(
        path.join(exp_path, 'best.hdf5'), monitor='val_acc', save_best_only=True, verbose=1)
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, verbose=1)
    csv_logger = keras.callbacks.CSVLogger(
        path.join(exp_path, 'log.csv'), append=True, separator=';')
    return [checkpoint, early_stop, csv_logger]


def train(model, word_encoder_model, lr=0.001, batch_size=64, epochs=50, patience=10, base_dir='experiments', **fit_args):
    optimizer = keras.optimizers.adam(lr=lr)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    exp_path = create_experiment_folder(
        base_dir, word_encoder_model, lr=lr, batch_size=batch_size)

    copy_called_file(exp_path)

    model.summary()

    with open(path.join(exp_path, 'config.txt'), 'a') as the_file:
        the_file.write(
            '\n'.join([str(x) for x in [lr, word_encoder_model.dropout_rate, batch_size, datetime.datetime.now()]]))

    history = model.fit(epochs=epochs,
                        batch_size=batch_size, callbacks=create_callbacks(exp_path, patience), **fit_args)

    best_acc = str(max(history.history['val_acc']))[:6]

    # append best acc
    deep_plots.from_keras_log(path.join(exp_path, 'log.csv'), exp_path)
    move(exp_path, exp_path + '_' + best_acc)


def load_csv(data_path=None, text_col='text', class_col='class', limit=None):
    X = []
    y = []

    with open(data_path) as csvfile:
        reader = csv.DictReader(csvfile)
        if limit is not None:
            reader = list(reader)[:limit]
        for row in reader:
            try:
                new_x = row[text_col]
                X.append(new_x)
                raw_y = row[class_col]
                y.append(raw_y)

            except Exception as e:
                print(e)

    return X, y


def process_save(X, y, tokenizer, proc_data_path, max_len=400, train=False, ngrams=None, limit_top_tokens=None):
    """Process text and save as Dataset
    """
    if train and limit_top_tokens is not None:
        tokenizer.apply_encoding_options(limit_top_tokens=limit_top_tokens)

    X_encoded = tokenizer.encode_texts(X)

    if ngrams is not None:
        X_encoded = tokenizer.add_ngrams(X_encoded, n=ngrams, train=train)

    X_padded = tokenizer.pad_sequences(
        X_encoded, fixed_token_seq_length=max_len)

    if train:
        ds = Dataset(X_padded,
                     y, tokenizer=tokenizer)
    else:
        ds = Dataset(X_padded, y)

    ds.save(proc_data_path)


def setup_data(X, y, tokenizer, proc_data_path, **kwargs):
    """Setup data

        Args:
            X: text data,
            y: data labels,
            tokenizer: A Tokenizer instance
            proc_data_path: Path for the processed data
    """
    # only build vocabulary once (e.g. training data)
    train = not tokenizer.has_vocab
    if train:
        tokenizer.build_vocab(X)

    process_save(X, y, tokenizer, proc_data_path,
                 train=train, **kwargs)
    return tokenizer


def split_data(X, y, ratio=(0.8, 0.1, 0.1)):
    """Splits data into a training, validation, and test set.

        Args:
            X: text data
            y: data labels
            ratio: the ratio for splitting. Default: (0.8, 0.1, 0.1)

        Returns:
            split data: X_train, X_val, X_test, y_train, y_val, y_test
    """
    assert(sum(ratio) == 1 and len(ratio) == 3)
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=ratio[0])
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, train_size=ratio[1])
    return X_train, X_val, X_test, y_train, y_val, y_test


def setup_data_split(X, y, tokenizer, proc_data_dir, **kwargs):
    """Setup data while splitting into a training, validation, and test set.

        Args:
            X: text data,
            y: data labels,
            tokenizer: A Tokenizer instance
            proc_data_dir: Directory for the split and processed data
    """
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # only build vocabulary on training data
    tokenizer.build_vocab(X_train)

    process_save(X_train, y_train, tokenizer, path.join(
        proc_data_dir, 'train.bin'), train=True, **kwargs)
    process_save(X_val, y_val, tokenizer, path.join(
        proc_data_dir, 'val.bin'), **kwargs)
    process_save(X_test, y_test, tokenizer, path.join(
        proc_data_dir, 'test.bin'), **kwargs)


def load_data_split(proc_data_dir):
    """Loads a split dataset

        Args:
            proc_data_dir: Directory with the split and processed data

        Returns:
            (Training Data, Validation Data, Test Data)
    """
    ds_train = Dataset.load(path.join(proc_data_dir, 'train.bin'))
    ds_val = Dataset.load(path.join(proc_data_dir, 'val.bin'))
    ds_test = Dataset.load(path.join(proc_data_dir, 'test.bin'))
    return ds_train, ds_val, ds_test
