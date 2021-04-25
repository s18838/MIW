#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 18:09:46 2021

This is example of Random tree usage

@author: taraskulyavets
"""

import argparse
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier 


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
    
    dataset = pd.read_csv(args.train_file)
    dataset = dataset.replace(np.nan, 0)
    
    if args.train_test_split  < 0.2:
        args.train_test_split = 0.2
    elif args.train_test_split > 0.8:
        args.train_test_split = 0.8
    
    responses = np.array(dataset[args.clasification_column])
    dataset = dataset.iloc[:,:-1]
    dataset_headers = list(dataset.columns)
    dataset = np.array(dataset).astype(np.float64)
    train_data, test_data, train_responses, test_responses = train_test_split(
        dataset, responses, test_size=args.train_test_split)
    rfc = RandomForestClassifier\
        (max_depth=args.max_depth, min_impurity_split=args.acceptable_impurity)
    rfc.fit(train_data, train_responses)
    
    predictions = rfc.predict(test_data)
    errors = abs(predictions - test_responses)
    print('Mean Absolute Error:', np.mean(errors))
    
    tree = rfc.estimators_[0]
    export_graphviz(tree, out_file='tree.dot', feature_names=dataset_headers, 
                    rounded=True, precision=1)
    os.system(f'dot -Tpng tree.dot -o tree.png')
    
    


def parse_arguments():
    """
    This function parses arguments from command line

    Returns
    -------
    argparse.Namespace
        Namespace containing all of argument from command line or their default values.

    """
    parser = argparse.ArgumentParser(description=("Backpropagation"))
    parser.add_argument("-t", "--train_file",
            action="store",
            required=True,
            help="Csv file with  data")
    parser.add_argument("-s", "--train_test_split",
            action="store",
            type=float,
            default=0.2,
            help="How many of datapoint will be used for tests (default and min 0.2 while max 0.8)")
    parser.add_argument("-c", "--clasification_column",
            action="store",
            required=True,
            help="Name of column in dataset with classification data")
    parser.add_argument("--max_depth",
            action="store",
            type=int,
            default=5,
            help="Maximum depth of tree, dafault value is 5")
    parser.add_argument("--acceptable_impurity",
            action="store",
            type=float,
            help="Level of impurity at which we no longer split nodes, default value 0",
            default=0.0)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())
    