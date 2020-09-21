# -*- coding: utf-8 -*-
"""
A simple test to test if the whole pipeline, as defined in the interface module, works.
"""

import unittest
import pandas as pd

from pywoe.interface import get_raw_data_to_woe_values_pipeline


class InterfaceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X = pd.DataFrame({
                'feat_1': [18, 78, 23, 45, 'T', 34],
                'feat_2': [567, 987, 123, 345, 'T', 876]
            },
            index=[0, 1, 2, 3, 4, 5]
        )
        cls.y = pd.Series(
            [0, 0, 1, 0, 1, 1],
            index=[0, 1, 2, 3, 4, 5]
        )

    def test_instantiation(self):
        get_raw_data_to_woe_values_pipeline()

    def test_fitting_on_small_dataset(self):
        pipeline = get_raw_data_to_woe_values_pipeline()
        pipeline.fit(
            X=self.X,
            y=self.y
        )
        print(
            pipeline.transform(X=self.X)
        )
