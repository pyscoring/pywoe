# -*- coding: utf-8 -*-
"""
A set of binner classes that function as a parameter to :class:`WoETransformer` and
define the binning algorithm.
"""

import pandas as pd

from typing import Dict, AnyStr
from abc import ABC, abstractmethod
from sklearn.tree import DecisionTreeClassifier
from pywoe.data_models.binning import BinningSpec
from pywoe.data_models.feature import Feature
from pywoe.feature_engineering.utils import retrieve_initial_bins_from_tree
from pywoe.constants import \
    DEFAULT_DECISION_TREE_CLASSIFIER_INIT_KWARGS, \
    DEFAULT_DECISION_TREE_CLASSIFIER_FIT_KWARGS


class AbstractBinner(ABC):
    """
    An abstract class defining the basic binner interface.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        A method that computes the bins from the raw data.

        :param X: the DataFrame where the columns are the variables to be binned
        :param y: the target which can be used to inform binning
        """
        pass

    def get_binning_spec(self) -> Dict[AnyStr, BinningSpec]:
        """
        A method that retrieves the computed/defined binnign spec.

        :return:
        """

        if self._spec is None:
            raise ValueError("Please fit the bins before proceeding with further steps!")

        else:
            return self._spec


class PreSpecifiedSpecBinner(AbstractBinner):
    """
    A binner that does not fit bins, but derives them from a pre-specified spec loaded during
    instantiation.
    """

    def __init__(self, spec: Dict[AnyStr, BinningSpec]):
        """
        :param spec: the binning spec used in the binner
        """

        self._spec = spec

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        In the case of the pre-specified binner, this is an empty method.
        """

        pass


class DecisionTreeBinner(AbstractBinner):
    """
    A binner that uses a decision tree classifier to determine bins.
    """

    def __init__(
            self,
            features: Dict[AnyStr, Feature],
            init_kwargs: Dict = DEFAULT_DECISION_TREE_CLASSIFIER_INIT_KWARGS,
            fit_kwargs: Dict = DEFAULT_DECISION_TREE_CLASSIFIER_FIT_KWARGS
    ):
        """
        :param features: the features for which we define the bins
        :param init_kwargs: the keyword argument dictionary that will be passed to the decision tree class
                            at instantiation
        :param fit_kwargs:  the keyword argument dictionary that will be passed to the decision tree class
                            when `tree.fit(...)` is called
        """

        self._features = features
        self._init_kwargs = init_kwargs
        self._fit_kwargs = fit_kwargs
        self._trees = {}
        self._initial_specs = {}
        self._spec = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        :param X: the DataFrame where the columns are the variables to be binned
        :param y: the target which can be used to inform binning
        """

        # TODO This piece of code should be parallelised
        for name in X.columns:
            numeric = pd.to_numeric(X[name], errors='coerce')
            self._trees[name] = DecisionTreeClassifier(**self._init_kwargs)
            self._trees[name].fit(
                numeric[numeric.notnull()],
                y[numeric.notnull()]
            )
            self._initial_specs[name] = retrieve_initial_bins_from_tree(
                self._features[name],
                self._trees[name],
                self._features[name].range.categorical_indicators
            )

        # For now, we skip the bin merging part
        self._spec = self._initial_specs
