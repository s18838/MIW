#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 22:30:40 2021

@author: taraskulyavets
"""

import argparse
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot  as plt


def create_network():
    """
    This method creates new network with specified architecture.

    Returns
    -------
    model : tensorflow.python.keras.engine.sequential.Sequential
        Model groups layers into an object with training and inference features.

    """
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(Conv2D(8, (3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(100, activation='softmax'))
    return model


def test_network(output_json, output_h5):
    """
    This method teaches and saves network.

    Parameters
    ----------
    output_json : str
        JSON file with network architecture.
    output_h5 : str
        HDF5 file with weights.

    Returns
    -------
    None.

    """
    (train_images, train_labels), (test_images, test_labels) = \
        tf.keras.datasets.cifar100.load_data(label_mode='fine')
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = train_images.reshape(-1, 32, 32, 3)
    test_images = test_images.reshape(-1, 32, 32, 3)
    train_labels = np_utils.to_categorical(train_labels, 100)
    test_labels = np_utils.to_categorical(test_labels, 100)

    model = create_network()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_images, train_labels, batch_size=32, epochs=10, \
                        validation_data=(test_images, test_labels))

    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend()
    plt.show()

    file = open(output_json, 'w')
    file.write(model.to_json())
    file.close()

    model.save(output_h5)


def main(args):
    """
    This is main function of this module, it runs neural network testing.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace containing all of argument from command line.

    Returns
    -------
    None.

    """
    test_network(args.output_json, args.output_h5)


def parse_arguments():
    """
    This function parses arguments from command line.

    Returns
    -------
    argparse.Namespace
        Namespace containing all of argument from command line or their default values.

    """
    parser = argparse.ArgumentParser(description=("AlexNet"))
    parser.add_argument("-o_json", "--output_json",
            action="store",
            default="model.json",
            help="Name of output JSON file with network architecture")
    parser.add_argument("-o_h5", "--output_h5",
            action="store",
            default="model.h5",
            help="Name of output HDF5 file with weights")
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())
