#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 18:09:46 2021

This is example of Random tree usage

@author: taraskulyavets
"""

import argparse
import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz, DecisionTreeClassifier
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
    if args.train_test_split < 0.2 or args.train_test_split > 0.8:
        print("Bad value for train_test_split, range is 0.2 - 0.8")
        sys.exit()

    dataset = pd.read_csv(args.train_file)

    x_data = dataset.loc[:, (dataset.columns != args.classification_column) \
                    & (dataset.columns != "Survey_id")]
    y_data = dataset[args.classification_column].to_numpy()
    dataset_headers = list(x_data.columns)
    x_data = x_data.fillna(0).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, \
                                                        test_size=args.train_test_split)


    dtc = DecisionTreeClassifier(max_depth=args.max_depth, \
                                 min_impurity_split=args.acceptable_impurity)
    dtc = dtc.fit(x_train, y_train)
    dtc_score = dtc.score(x_test, y_test)


    export_graphviz(dtc, out_file="decision_tree.dot", feature_names=dataset_headers, \
                    rounded=True, precision=1, filled=True)
    os.system("dot -Tpng decision_tree.dot -o decision_tree.png")


    rfc = RandomForestClassifier(n_estimators=args.estimators, max_depth=args.max_depth, \
                                 min_impurity_split=args.acceptable_impurity)
    rfc.fit(x_train, y_train)
    rfc_score = rfc.score(x_test, y_test)

    file = open('result.txt', 'w')
    file.write(f'Decisions tree score = {dtc_score}\n')
    file.write(f'Random forest score = {rfc_score}\n')
    file.close()


def parse_arguments():
    """
    This function parses arguments from command line

    Returns
    -------
    argparse.Namespace
        Namespace containing all of argument from command line or their default values.

    """
    parser = argparse.ArgumentParser(description=("CART example"))
    parser.add_argument("-t", "--train_file",
            action="store",
            required=True,
            help="Csv file with  data")
    parser.add_argument("-s", "--train_test_split",
            action="store",
            type=float,
            default=0.2,
            help="How many of datapoint will be used for tests \
                (default and min 0.2 while max 0.8)")
    parser.add_argument("-c", "--classification_column",
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
            default=0.0,
            help="Level of impurity at which we no longer split nodes, default value 0")
    parser.add_argument('--estimators',
            type=int,
            default=9,
            help='Number of estimators for random forest')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())
    