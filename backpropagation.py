#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 17:10:02 2021

This is NeuralLayer script file

@author: taraskulyavets
"""

import argparse
import pandas as pd
import numpy as np


class NeuralLayer:
    """
    This is Neural Layer class
    """

    def __init__(self, weights, bias, use_bipolar = False):
        """
        This is Neural Layer constructor

        Parameters
        ----------
        weights : numpy.ndarray
            Layer weights.
        bias : numpy.ndarray
            Layer bias.
        use_bipolar : TYPE, optional
            A flag used to define which activation function to use. The default is False.

        Returns
        -------
        None.

        """
        self.weights = weights
        self.bias = bias
        self.use_bipolar = use_bipolar
        self.outputs = None

    def activation_function(self, u_prim):
        """
        This method chooses which activation function will be used

        Parameters
        ----------
        u_prim : numpy.ndarray
            List of layer outputs.

        Returns
        -------
        numpy.ndarray
            Layer`s activation function output.

        """
        if self.use_bipolar:
            return 2 / (1 + np.exp(-u_prim)) - 1
        return 1 / (1 + np.exp(-u_prim))

    def feed_forward(self, inputs):
        """
        This method sends inputs through layer neurons

        Parameters
        ----------
        inputs : numpy.ndarray
            Input data.

        Returns
        -------
        numpy.ndarray
            Layer output values

        """
        u = np.matmul(self.weights, inputs)
        u_prim = u + self.bias
        y = self.activation_function(u_prim)
        self.outputs = y
        return y

    @staticmethod
    def generate_random_layer(input_size, layer_size, use_bipolar):
        """
        This method creates new instance of Neural Layer

        Parameters
        ----------
        input_size : int
            Size of input.
        layer_size : int
            Size of layer.
        use_bipolar : bool
            A flag used to define which activation function to use.

        Returns
        -------
        NeuralLayer
            New instance of Neural Layer.

        """
        weights = np.random.rand(layer_size, input_size) * 2 - 1
        bias = np.random.rand(layer_size) * (-1)
        return NeuralLayer(weights, bias, use_bipolar)


class NeuralNetwork:
    """
    This is Neural Network class
    """

    def __init__(self, layers, learning_factor = 0.1, use_bipolar = False):
        """
        This is Neural Network constructor

        Parameters
        ----------
        layers : list
            Neural network layers.
        learning_factor : float, optional
            Learning factor, define how quickly network will train. The default is 0.1.
        use_bipolar : bool, optional
            A flag used to define which activation function to use. The default is False.

        Returns
        -------
        None.

        """
        self.layers = layers
        self.learning_factor = learning_factor
        self.use_bipolar = use_bipolar

    def derivative_function(self, output):
        """
        This method chooses which derivative function will be used

        Parameters
        ----------
        output : numpy.ndarray
            List of layer outputs.

        Returns
        -------
        numpy.ndarray
            Layer`s derivative function output.

        """
        if self.use_bipolar:
            return 1 - output ** 2
        return output * (1 - output)

    def feed_forward(self, inputs):
        """
        This method sends inputs through Neural Network

        Parameters
        ----------
        inputs : numpy.ndarray
            Input data.

        Returns
        -------
        numpy.ndarray
            Output values, which are used to predict class.

        """
        for layer in self.layers:
            inputs = layer.feed_forward(inputs)
        return inputs

    def feed_backward(self, expected, row):
        """
        This method changes weigths accordingly to error

        Parameters
        ----------
        expected : list
            List with values, which define which class we are expected.
        row : numpy.ndarray
            Training input.

        Returns
        -------
        None.

        """
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i != (len(self.layers) - 1):
                errors =  np.matmul(self.layers[i + 1].delta, self.layers[i + 1].weights)
            else:
                errors = expected - layer.outputs
            layer.delta = errors * self.derivative_function(layer.outputs)
            if i == 0:
                inputs = row
            else:
                inputs = self.layers[i - 1].outputs
            layer.weights += np.atleast_2d(layer.delta).T * inputs * self.learning_factor
            layer.bias += layer.delta * self.learning_factor

    def train(self, train, output_classes, epochs):
        """
        This method trains Neural Network and print error

        Parameters
        ----------
        train : numpy.ndarray
            Dataset with rows to train.
        output_classes : list
            List with all possible output class.
        epochs : int
            Number of epochs.

        Returns
        -------
        None.

        """
        sum_error = 0
        for _ in range(epochs):
            for row in train:
                inputs = row[:-1].astype(np.float64)
                outputs = self.feed_forward(inputs)
                expected = [0 for i in range(len(output_classes))]
                expected[output_classes.index(row[-1])] = 1
                sum_r = sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                print(f'{row} row error - {sum_r}')
                sum_error += sum_r
                self.feed_backward(expected, inputs)
        print(f'mean test error - {sum_error / (len(train) * epochs)}')

    @staticmethod
    def create_network(input_size, layers_sizes, learning_factor, use_bipolar):
        """
        This method creates new instance of Neural Network

        Parameters
        ----------
        input_size : int
            Size of input.
        layers_sizes : list
            Neural network layers sizes.
        learning_factor : float
            Learning factor, define how quickly network will train.
        use_bipolar : bool
            A flag used to define which activation function to use.

        Returns
        -------
        NeuralNetwork
            New instance of Neural Network.

        """
        layers = []
        for layer_size in layers_sizes:
            layers.append(NeuralLayer.generate_random_layer(input_size, layer_size, use_bipolar))
            input_size = layer_size
        return NeuralNetwork(layers, learning_factor, use_bipolar)


def test_network(file_name, test_split, learning_factor, hidden_size, use_bipolar, epochs):
    """
    This function reads datafrom file and train neural network

    Parameters
    ----------
    file_name : str
        Name of input file with dataset.
    test_split : float
        Percent of rows to test.
    learning_factor : float
        Learning factor, define how quickly network will train.
    hidden_size : int
        Size of hidden layer.
    use_bipolar : bool
        A flag used to define which activation function to use.
    epochs : int
        Number of epochs.

    Returns
    -------
    None.

    """
    layers_sizes = [hidden_size]
    dataset = pd.read_csv(file_name)
    dataset = dataset.iloc[:,1:]
    train = dataset.sample(frac = (1 - test_split))
    test = dataset.drop(train.index)
    train = train.to_numpy()
    test = test.to_numpy()
    input_size = len(train[0]) - 1
    output_classes = list(set([row[-1] for row in np.array(dataset)]))
    output_size = len(output_classes)
    layers_sizes.append(output_size)
    network = NeuralNetwork.create_network(input_size, layers_sizes, learning_factor, use_bipolar)
    network.train(train, output_classes, epochs)


def main(args):
    """
    This is main function of this module, it runs neural network testing

    Parameters
    ----------
    args : argparse.Namespace
        Namespace containing all of argument from command line.

    Returns
    -------
    None.

    """
    test_network(args.input, args.test_split, args.learning_factor,
                args.hidden, args.bipolar, args.epochs)


def parse_arguments():
    """
    This function parses arguments from command line

    Returns
    -------
    argparse.Namespace
        Namespace containing all of argument from command line or their default values.

    """
    parser = argparse.ArgumentParser(description=("Backpropagation"))
    parser.add_argument("-i", "--input",
            action="store",
            default="Iris.csv",
            help="Input file for training network")
    parser.add_argument("--test_split",
            action="store",
            type=float,
            default=0.3,
            help="What part of data should be used for validation (default 0.3)")
    parser.add_argument("-e", "--learning_factor",
            action="store",
            help="Learning factor",
            default=0.1)
    parser.add_argument("--bipolar",
            action="store_true",
            help="If set use bipolar function otherwise unipolar")
    parser.add_argument("--hidden",
            action="store",
            type=int,
            help="Size of hidden layer",
            default=4)
    parser.add_argument("--epochs",
            action="store",
            type=int,
            help="Number of epochs",
            default=100)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())
