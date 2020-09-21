# -*- coding: utf-8 -*-
"""
Weigt-of-Evidence `sklearn` transformer class for use in `sklearn` pipelines.
"""

import pandas as pd
import numpy as np

from typing import Dict, AnyStr, Type
from sklearn.base import BaseEstimator, TransformerMixin
from pywoe.data_models.woe import WoESpec, WoEBin
from pywoe.feature_engineering.binning import AbstractBinner
from pywoe.constants import NUMERIC_ACCURACY
from pywoe.feature_engineering.utils import get_mask_from_range


def _compute_woe(
        all_event_count: int,
        all_non_event_count: int,
        bin_event_count: int,
        bin_non_event_count: int
) -> float:
    """
    A method that computes the Weight-of-Evidence value for a bin.

    :param all_event_count: the number of events (e.g. total number of bads)
    :param all_non_event_count: the number of non-events (e.g. total number of goods)
    :param bin_event_count: the number of events in the bin (e.g. bads in the bin)
    :param bin_non_event_count: the number of non-events in the bin (e.g. goods in the bin)
    :return: the Weight-of-Evidence value
    """

    bin_event_prcnt = bin_event_count / (all_event_count + NUMERIC_ACCURACY)
    bin_non_event_prcnt = bin_non_event_count / (all_non_event_count + NUMERIC_ACCURACY)
    return np.log(bin_event_prcnt / (bin_non_event_prcnt + NUMERIC_ACCURACY))


def _compute_iv(
        all_event_count: int,
        all_non_event_count: int,
        bin_event_count: int,
        bin_non_event_count: int
):
    """
    A method that computes the Information Value for a bin.

    :param all_event_count: the number of events (e.g. total number of bads)
    :param all_non_event_count: the number of non-events (e.g. total number of goods)
    :param bin_event_count: the number of events in the bin (e.g. bads in the bin)
    :param bin_non_event_count: the number of non-events in the bin (e.g. goods in the bin)
    :return: the Information Value
    """

    bin_event_prcnt = bin_event_count / (all_event_count + NUMERIC_ACCURACY)
    bin_non_event_prcnt = bin_non_event_count / (all_non_event_count + NUMERIC_ACCURACY)

    return (bin_event_prcnt - bin_non_event_prcnt) * \
        np.log(bin_event_prcnt / (bin_non_event_prcnt + NUMERIC_ACCURACY))


class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Weigt-of-Evidence `sklearn` transformer class for use in `sklearn` pipelines.
    """

    def __init__(
            self,
            binner: Type[AbstractBinner] = None,
            woe_spec: Dict[AnyStr, WoESpec] = None
    ):
        """

        :param woe_spec: in case we want to load a pre-defined WoE specification, we can provide
                         this parameter with a value
        :param binner: the binner that will be used to define bins over which WoE will be computed
        """

        if woe_spec is not None:
            self.woe_spec = woe_spec

        elif binner is not None:
            self._binner = binner
            self.woe_spec = None

        else:
            raise ValueError("Either the WoE specification or the binner has to be provided.")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        :param X: the dataset to be WoE-transformed
        :param y: the target to serve in computing WoE imputation values
        :return: self
        """

        # TODO some checking on `X` and `y` to validate them should be done
        if self.woe_spec is None:
            self.woe_spec = {}
            self._binner.fit(X, y)
            binning_spec = self._binner.get_binning_spec()
            event = y.astype(bool)
            non_event = ~(y.astype(bool))

            for name, spec in binning_spec.items():
                all_event_count = event.sum()
                all_non_event_count = non_event.sum()
                woe_bins = set()

                # Impute with the right WoE value
                for bin in spec.bins:
                    bin_mask = get_mask_from_range(X[name], bin)
                    bin_event_count = event[bin_mask].sum()
                    bin_non_event_count = non_event[bin_mask].sum()
                    woe_bins.add(
                        WoEBin(
                            bin_event_count=bin_event_count,
                            bin_non_event_count=bin_non_event_count,
                            woe=_compute_woe(
                                all_event_count,
                                all_non_event_count,
                                bin_event_count,
                                bin_non_event_count
                            ),
                            iv=_compute_iv(
                                all_event_count,
                                all_non_event_count,
                                bin_event_count,
                                bin_non_event_count
                            ),
                            bin=bin
                        )
                    )

                self.woe_spec[name] = WoESpec(
                    feature=spec.feature,
                    bins=woe_bins
                )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a WoE transformation, replacing raw feature values with corresponding WoE bin values.

        :param X: the `pandas` DataFrame to be imputed with WoE values
        :return: a transformed DataFrame with WoE values instead of raw input features
        """

        # A copy to work on when transforming columns
        X_copy = X[list(self.woe_spec.keys())].copy(deep=True)

        if self.woe_spec is None:
            raise ValueError("Please fit the transformer before applying it!")

        for name, spec in self.woe_spec.items():

            # Impute with the right WoE value
            for woe_bin in spec.bins:
                X_copy.loc[get_mask_from_range(X[name], woe_bin.bin), name] = woe_bin.woe

        print(self.woe_spec)
        print(X_copy)
        return X_copy
