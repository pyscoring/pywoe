# -*- coding: utf-8 -*-
"""
A simple interface to be used.
"""

from pywoe.feature_engineering.validator import FeatureValidator
from pywoe.feature_engineering.binning import DecisionTreeBinner
from pywoe.feature_engineering.woe import WoETransformer
from sklearn.pipeline import Pipeline


def get_raw_data_to_woe_values_pipeline() -> Pipeline:
    """
    A simple interface into the library, good for quickly starting with the default settings.

    :return: an instantiated pipeline for automated WoE binning
    """

    feature_validator = FeatureValidator()
    binner = DecisionTreeBinner(feature_validator=feature_validator)
    woe_transformer = WoETransformer(binner=binner)

    # Keep in mind `binner` is not an `sklearn` object, it is a parameter to `woe_transformer`,
    # so it's not used in the pipeline.
    return Pipeline([
        ('validator', feature_validator),
        ('woe_transformer', woe_transformer)
    ])

