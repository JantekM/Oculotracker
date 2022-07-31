import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import Morphology
import PrepareDataset
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import winsound


def eval_param(param, val):
    custom = Morphology.defaultOptions()
    custom[param] = val
    model, x_train, x_test, y_train, y_test = prepare_custom_model(custom)
    return check_model(model, x_train, y_train, x_test, y_test, times = 10)


def prepare_custom_model(custom: dict = None):
    if custom is None:
        custom = Morphology.defaultOptions()

    PrepareDataset.prepare_dataset(filename="custom", debug = False, custom=custom)
    x,y = PrepareDataset.load_dataset('custom.npz')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(x_train)

    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(units=256, input_shape=(x_train.shape[1],), activation='relu'),
        #layers.Dropout(rate = 0.05),
        #layers.Dense(units=2048, activation='relu'),
        #layers.Dense(units=128, activation='relu'), ##
        #layers.Dense(units=128, activation='relu'),
        layers.Dense(units=128, activation='relu'),
        #layers.Dropout(rate = 0.05),
        layers.Dense(units=128, activation='relu'),
        #layers.Dropout(rate = 0.05),
        layers.Dense(units=32, activation='relu'),
        layers.Dense(units=2)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=00.005),
        loss='mean_squared_error')

    return model, x_train, x_test, y_train, y_test

def fit_model(model, epochs: int, x_train, y_train):
    hist= model.fit(
    x_train,
    y_train,
    epochs=epochs,
    validation_split = 0.1)
    return hist, model

def acc_of_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    diff = predictions - y_test
    err = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    return np.median(err)

def check_model(model, x_train, y_train, x_test, y_test, times: int):
    results = np.empty(shape = (times,))
    _, model = fit_model(model, 10, x_train, y_train)
    _, model = fit_model(model, 300, x_train, y_train)

    for i in range(times):
        _, model = fit_model(model, 3, x_train, y_train)
        results[i] = acc_of_model(model, x_test, y_test)
    return np.mean(results), np.std(results)


if __name__ == "__main__":
    errs = []
    sds = []
    params = ['minArea', 'maxArea', 'minInertia', 'maxInertia', 'threshold1', 'threshold2', 'threshold3', 'threshold4', 'tophatSize', 'blobErode', 'blobDilate', 'blobBlur']
    vals = [9, 9000, 0.14, 0.95, 23, 31, 61, 74, 9, 1, 3, 4]
    for param, val in zip(params, vals):
        err, sd = eval_param(param, val)
        errs.append(83 - err)
        sds.append(sd)
        winsound.Beep(400, 400)
    print(errs)
    print(sds)
    winsound.Beep(800, 2700)
    pass