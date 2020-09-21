# -*- coding: utf-8 -*-
"""
A feature range checker/validator that can be used as part of an `sklearn` pipeline.
It implements the :class:`TransformerMixin` interface, even though it does _not_ transform
the data apart from type checking and conversion where needed.
"""

import pandas as pd

from typing import Dict
from sklearn.base import BaseEstimator, TransformerMixin
from pywoe.data_models.feature import Feature
from pywoe.feature_engineering.utils import retrieve_feature_definition
from pywoe.constants import NUMERIC_ACCURACY


class FeatureValidator(BaseEstimator, TransformerMixin):
    """
    A feature range checker/validator that can be used as part of an `sklearn` pipeline.
    It implements the :class:`TransformerMixin` interface, even though it does _not_ transform
    the data apart from type checking and conversion where needed.
    """

    def __init__(
            self,
            feature_spec: Dict[str, Feature] = None
    ):
        """
        Feature validator instantiation that can be used to load a serialised feature validator via set of
        class instances of :class:`Feature`.

        :param feature_spec: a specification of the features that can be provided; if it's provided, `fit`
                             method won't change it. Can be used to retrieve a serialised feature validator.
        """

        if feature_spec is not None:
            self.feature_spec = feature_spec

        else:
            self.feature_spec = None

    def fit(self, X: pd.DataFrame, *args, **kwargs):
        """
        Determines data ranges in provided training data and builds a feature validation specification
        from that.

        :param X: the data to be used
        """

        if self.feature_spec is None:
            self.feature_spec = {}

            for col in X.columns:
                self.feature_spec[col] = retrieve_feature_definition(X[col])

        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Throws a :any:`ValueError` if there are unrecognised char/numeric values
        in the new inference-bound data.

        :param X: the inference-bound data to be used
        """

        # A copy to work on when transforming types
        X_copy = X[list(self.feature_spec.keys())].copy(deep=True)

        if self.feature_spec is None:
            raise ValueError("Please fit the transformer before applying it!")

        for name, feat in self.feature_spec.items():
            numeric = pd.to_numeric(X[name], errors='coerce')
            char_values = frozenset(X[name][numeric.isnull()].values)
            set_difference = char_values - feat.range.categorical_indicators

            if feat.range.numeric_range_start - numeric.min() > NUMERIC_ACCURACY:
                raise ValueError("The feature `{name}` is outside its range (min={min})".format(
                    name=name,
                    min=feat.range.numeric_range_start
                ))

            elif numeric.max() - feat.range.numeric_range_end > NUMERIC_ACCURACY:
                raise ValueError("The feature `{name}` is outside its range (max={max})".format(
                    name=name,
                    min=feat.range.numeric_range_end
                ))

            elif len(set_difference) > 0:
                raise ValueError("The feature `{name}` has unrecognised char values: {vals}".format(
                    name=name,
                    vals=list(set_difference)
                ))

            else:
                X_copy.loc[numeric.notnull(), name] = numeric[numeric.notnull()]
                X_copy.loc[numeric.isnull(), name] = X_copy.loc[numeric.isnull(), name].astype(str)

        return X_copy
