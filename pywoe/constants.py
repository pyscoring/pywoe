# -*- coding: utf-8 -*-
"""
Constants used throughout the package.
"""

INFINITY = 1e20
"""
A value for infinity.
"""

NUMERIC_ACCURACY = 1e-6
"""
The minimun difference between floats needed to deem them equal.
"""

DEFAULT_DECISION_TREE_CLASSIFIER_INIT_KWARGS = {
    "criterion": "gini",
    "max_depth": 5,
    "min_samples_leaf": 0.1
}
"""
The keyword argument dictionary that will be passed to the decision tree class
at instantiation, by default if the user does not override.
"""

DEFAULT_DECISION_TREE_CLASSIFIER_FIT_KWARGS = {}
"""
The keyword argument dictionary that will be passed to the decision tree class
at the call to `tree.fit(...)` method, by default if the user does not override.
"""

P_VALUE_THRESHOLD = 0.05
"""
The p-value threshold when judging if a stat. test succeeds.
"""
