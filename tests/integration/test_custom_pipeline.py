# -*- coding: utf-8 -*-
"""
A simple test that checks whether a custom pipeline works well on one of the standard `sklearn`
datasets.
"""

import cattr
import unittest
import pprint

from pywoe.feature_engineering.validator import FeatureValidator
from pywoe.feature_engineering.binning import DecisionTreeBinner
from pywoe.feature_engineering.woe import WoETransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer


class CustomPipelineTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_breast_cancer(return_X_y=True, as_frame=True)
        cls.feature_validator = FeatureValidator()
        cls.binner = DecisionTreeBinner(
            feature_validator=cls.feature_validator,
            init_kwargs={
                "criterion": "gini",
                "max_depth": 3,
                "min_samples_leaf": 0.2
            }
        )
        cls.woe_transformer = WoETransformer(binner=cls.binner)
        cls.pipeline = Pipeline([
            ('validator', cls.feature_validator),
            ('woe_transformer', cls.woe_transformer)
        ])

    def test_fitting(self):
        self.pipeline.fit(self.X, self.y)
        print(self.pipeline.transform(self.X))
        print(self.pipeline['woe_transformer'].woe_spec['mean radius'].bins)
