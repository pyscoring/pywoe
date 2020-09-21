# -*- coding: utf-8 -*-
"""
A set of binner classes that function as a parameter to :class:`WoETransformer` and
define the binning algorithm.
"""

import pandas as pd
import numpy as np

from typing import Dict, AnyStr, Callable
from abc import ABC, abstractmethod
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats import proportion
from pywoe.data_models.base import Range
from pywoe.data_models.binning import BinningSpec
from pywoe.feature_engineering.validator import FeatureValidator
from pywoe.feature_engineering.utils import \
    retrieve_initial_bins_from_tree, \
    get_mask_from_range
from pywoe.constants import \
    DEFAULT_DECISION_TREE_CLASSIFIER_INIT_KWARGS, \
    DEFAULT_DECISION_TREE_CLASSIFIER_FIT_KWARGS, \
    P_VALUE_THRESHOLD


def _proportion_z_test_returning_p_value(
        event_count_bin_1: int,
        event_count_bin_2: int,
        obs_count_bin_1: int,
        obs_count_bin_2: int
) -> float:
    """
    A wrapper around the `statsmodels` proportions z-test to only return p-value.

    :param event_count_bin_1: the total number of events (e.g. bads) in bin 1
    :param event_count_bin_2: the total number of events (e.g. bads) in bin 2
    :param obs_count_bin_1: the total number of observatons in bin 1
    :param obs_count_bin_2: the total number of observations in bin 2
    :return: a single float, the p-value of the test
    """

    _, p_value = proportion.proportions_ztest(
        np.array([event_count_bin_1, event_count_bin_2]),
        np.array([obs_count_bin_1, obs_count_bin_2])
    )
    return p_value


def _iteratively_merge_bins(
        x: pd.Series,
        y: pd.Series,
        binning_spec: BinningSpec,
        stat_test: Callable = _proportion_z_test_returning_p_value,
        p_value_threshold: float = P_VALUE_THRESHOLD
) -> BinningSpec:
    """
    TODO fix the method

    :param x: the variable being binned
    :param y: the target variable used to determine which bins to merge
    :param binning_spec: the binning specification for the feature; it will be iteratively merged
    :param stat_test: a function that returns a single value, p-value of a statistical test
    :param p_value_threshold: the threshold to decide if the null hypothesis is rejected
    :return: a binning specification with similar contiguous bins merged
    """

    bins_merged_at_iteration = True
    new_binning = binning_spec.bins

    while bins_merged_at_iteration:
        old_binning = new_binning
        new_binning = set()
        bins_ordered = list(old_binning)
        bin_masks = [get_mask_from_range(x, bin) for bin in bins_ordered]
        bin_event_counts = [y[mask].sum() for mask in bin_masks]
        bin_sizes = [mask.sum() for mask in bin_masks]
        bad_rates = [events / size for events, size in zip(bin_event_counts, bin_sizes)]

        # Sort bins according to bad rates.
        sorted_indexes = np.argsort(bad_rates)
        i = 1

        # Merge contiguous bins that have bad rates that aren't stat. sign. different.
        while i < len(sorted_indexes):
            prev_idx = sorted_indexes[i - 1]
            idx = sorted_indexes[i]
            p_value = stat_test(
                bin_event_counts[prev_idx],
                bin_event_counts[idx],
                bin_sizes[prev_idx],
                bin_sizes[idx]
            )

            if p_value >= p_value_threshold:
                print(
                    'merging bins ({}, {}) - ({}, {})'.format(
                        bins_ordered[prev_idx].numeric_range_start,
                        bins_ordered[prev_idx].numeric_range_end,
                        bins_ordered[idx].numeric_range_start,
                        bins_ordered[idx].numeric_range_end
                    )
                )
                new_binning.add(
                    Range(
                        numeric_range_start=min(
                            bins_ordered[prev_idx].numeric_range_start,
                            bins_ordered[idx].numeric_range_start
                        ),
                        numeric_range_end=max(
                            bins_ordered[prev_idx].numeric_range_end,
                            bins_ordered[idx].numeric_range_end
                        ),
                        categorical_indicators=frozenset.union(
                            bins_ordered[prev_idx].categorical_indicators,
                            bins_ordered[idx].categorical_indicators
                        )
                    )
                )
                i += 2

                # If we have skipped past the last bin, add it.
                if i == len(sorted_indexes):
                    new_binning.add(bins_ordered[sorted_indexes[-1]])

            else:
                new_binning.add(bins_ordered[prev_idx])
                i += 1

        if len(old_binning) == len(new_binning):
            bins_merged_at_iteration = False

        elif len(old_binning) > len(new_binning):
            bins_merged_at_iteration = True

        else:
            raise RuntimeError()

    return BinningSpec(
        feature=binning_spec.feature,
        bins=new_binning
    )


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
            feature_validator: FeatureValidator,
            init_kwargs: Dict = DEFAULT_DECISION_TREE_CLASSIFIER_INIT_KWARGS,
            fit_kwargs: Dict = DEFAULT_DECISION_TREE_CLASSIFIER_FIT_KWARGS
    ):
        """
        :param feature_validator: a fitted instance of a feature validator
        :param init_kwargs: the keyword argument dictionary that will be passed to the decision tree class
                            at instantiation
        :param fit_kwargs:  the keyword argument dictionary that will be passed to the decision tree class
                            when `tree.fit(...)` is called
        """

        self._feature_validator = feature_validator
        self._init_kwargs = init_kwargs
        self._fit_kwargs = fit_kwargs
        self._trees = {}
        self._initial_specs = {}
        self._spec = None

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            stat_test: Callable = _proportion_z_test_returning_p_value,
            p_value_threshold: float = P_VALUE_THRESHOLD
    ) -> None:
        """
        :param X: the DataFrame where the columns are the variables to be binned
        :param y: the target which can be used to inform binning
        :param stat_test: the statistical test used. It's a method that expects 4 values:
                            * the number of events (e.g. bads) in bin 1
                            * the number of events (e.g. bads) in bin 2
                            * total number of datapoints in bin 1
                            * total number of datapoints in bin 2
        :param p_value_threshold: the threshold to decide if the null hypothesis is rejected when comparing
                                  whether two bins have the same event (e.g. bad) rate
        """

        # TODO This piece of code should be parallelised
        for name in X.columns:
            numeric = pd.to_numeric(X[name], errors='coerce')
            self._trees[name] = DecisionTreeClassifier(**self._init_kwargs)
            self._trees[name].fit(
                np.expand_dims(numeric[numeric.notnull()], 1),
                y[numeric.notnull()]
            )
            self._initial_specs[name] = retrieve_initial_bins_from_tree(
                self._feature_validator.feature_spec[name],
                self._trees[name],
                self._feature_validator.feature_spec[name].range.categorical_indicators
            )

        # Finally, we go through bins, including the categorical ones, and merge the ones that are not
        # stat. sign. different from neighbouring ones.
        # TODO This piece of code should be parallelised
        self._spec = self._initial_specs
        """{
            name: _iteratively_merge_bins(
                X[name],
                y,
                self._initial_specs[name],
                stat_test=stat_test,
                p_value_threshold=p_value_threshold
            )

            for name in self._feature_validator.feature_spec.keys()
        }"""
