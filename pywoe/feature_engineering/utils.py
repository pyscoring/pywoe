# -*- coding: utf-8 -*-
"""
Utility methods used throughout feature engineering/preprocessing classes.
"""

import pandas as pd

from typing import List, FrozenSet
from sklearn.tree import DecisionTreeClassifier
from pywoe.data_models.base import Range
from pywoe.data_models.feature import Feature
from pywoe.data_models.binning import BinningSpec
from pywoe.constants import NUMERIC_ACCURACY


def retrieve_feature_definition(ser: pd.Series) -> Feature:
    """
    Retrieves a feature definition from a `pandas` Series.

    :param ser: the `pandas` Series to retrieve the feature definition from
    :return: a feature definition
    """

    numeric = pd.to_numeric(ser, errors='coerce')
    categorical = ser[numeric.isnull()]

    return Feature(
        name=ser.name,
        range=Range(
            numeric_range_start=numeric.min() - NUMERIC_ACCURACY,
            numeric_range_end=numeric.max() + NUMERIC_ACCURACY,
            categorical_indicators=frozenset(categorical.values)
        )
    )


def get_mask_from_range(ser: pd.Series, bin: Range) -> pd.Series:
    """
    Transforms a range object to a mask on a `pandas` Series.

    :param ser: the `pandas` Series to be turned into a mask
    :param bin: the range object that defines the range the mask should pick out
    :return: a boolean mask to be applied to filter out just the bit of the dataframe that falls in
             the range
    """

    if bin.numeric_range_start is not None:
        return (
                (pd.to_numeric(ser, errors='coerce') > bin.numeric_range_start) & \
                (pd.to_numeric(ser, errors='coerce') <= bin.numeric_range_end)
            ) | (
                ser.isin(bin.categorical_indicators)
            )

    else:
        return ser.isin(bin.categorical_indicators)


def retrieve_initial_bins_from_tree(
        feature: Feature,
        tree: DecisionTreeClassifier,
        categorical_indicators: FrozenSet
) -> BinningSpec:
    """
    Converts a decision tree to an initial binning spec further refined in binning process.
    This initial bin set does not merge similar bins with categorical indicators; that's done later.
    See the `binning` module for more.

    :param feature: the feature that's being binned
    :param tree: a fitted `sklearn` decision tree classifier
    :param categorical_indicators: the categorical indicator set to be used
    :return: a :class:`BinningSpec` object specifying binning for a feature
    """

    stack = [0]
    bin_thresholds = [feature.range.numeric_range_start, feature.range.numeric_range_end]

    while len(stack) > 0:
        node_id = stack.pop()

        if not _is_leaf(tree, node_id):
            left_child = tree.tree_.children_left[node_id]
            right_child = tree.tree_.children_right[node_id]
            stack.append(right_child)
            stack.append(left_child)

            if _is_leaf(tree, left_child) or _is_leaf(tree, right_child):
                bin_thresholds.append(tree.tree_.threshold[node_id])

    sorted_thresholds = sorted(list(set(bin_thresholds)))

    return _get_spec(
        feature,
        sorted_thresholds,
        categorical_indicators
    )


def _get_spec(
        feature: Feature,
        sorted_thresholds: List,
        categorical_indicators: FrozenSet
) -> BinningSpec:
    """
    Creates a binning spec from provided data.

    :param feature: the feature that's being binned
    :param sorted_thresholds: the numeric part thresholds, sorted and unique
    :param categorical_indicators: the categorical indicator set to be used
    :return:
    """

    list_of_sets_required = [
        set([
            Range(
                numeric_range_start=sorted_thresholds[i],
                numeric_range_end=threshold
            )
        ])

        for i, threshold in enumerate(sorted_thresholds[1:])
    ] + [
        set([
            Range(
                categorical_indicators=set([char])
            )
        ])

        for char in categorical_indicators
    ]

    return BinningSpec(
        feature=feature,
        bins=set.union(*list_of_sets_required)
    )


def _is_leaf(tree: DecisionTreeClassifier, node_id: int) -> bool:
    """
    Determines if a tree node is a leaf.

    :param tree: an `sklearn` decision tree classifier object
    :param node_id: an integer identifying a node in the above tree
    :return: a boolean `True` if the node is a leaf, `False` otherwise
    """

    return tree.tree_.children_left[node_id] == tree.tree_.children_right[node_id]
