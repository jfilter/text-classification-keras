import csv
import datetime
import inspect
import os
import pathlib
from os import path
from shutil import copyfile, move

import deep_plots
import keras

from .data import Dataset
from .utils.format import to_fixed_digits

# Python 3.5+ only


def create_experiment_folder(base_dir, model, lr, batch_size):
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


def train(fit_args, model, word_encoder_model, lr=0.001, batch_size=64, epochs=50, patience=10, base_dir='experiments'):
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

    history = model.fit(**fit_args, epochs=epochs,
                        batch_size=batch_size, callbacks=create_callbacks(exp_path, patience))

    best_acc = str(max(history.history['val_acc']))[:6]

    # append best acc
    deep_plots.from_keras_log(path.join(exp_path, 'log.txt'), exp_path)
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


def setup_data(tokenizer, proc_data_path, max_len=400, load_csv_args={}):
    X, y = load_csv(**load_csv_args)

    # onyl build vocab on training data
    if not tokenizer.has_vocab:
        tokenizer.build_vocab(X)

    X_encoded = tokenizer.encode_texts(X)
    X_padded = tokenizer.pad_sequences(
        X_encoded, fixed_token_seq_length=max_len)

    ds = Dataset(X_padded,
                 y, tokenizer=tokenizer)

    ds.save(proc_data_path)
    return tokenizer
