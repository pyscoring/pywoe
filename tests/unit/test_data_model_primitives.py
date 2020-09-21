# -*- coding: utf-8 -*-
"""
Data model primitive building block tests. Mainly checks that validation works as expected.
"""

import unittest

from pywoe.constants import INFINITY
from pywoe.data_models.base import Range
from pywoe.data_models.feature import Feature
from pywoe.data_models.woe import WoEBin


class DataModelPrimitivesTests(unittest.TestCase):

    def test_passing_no_argument_works(self):

        Range(
            categorical_indicators={"a", "b", "c"}
        )

    def test_passing_no_argument_does_not_work_right(self):

        with self.assertRaises(ValueError):
            Range(
                numeric_range_end=1.,
                categorical_indicators={"a", "b", "c"}
            )

    def test_passing_no_argument_does_not_work_left(self):

        with self.assertRaises(ValueError):
            Range(
                numeric_range_start=1.,
                categorical_indicators={"a", "b", "c"}
            )

    def test_passing_above_infinity_does_not_work(self):

        with self.assertRaises(ValueError):
            Range(
                numeric_range_start=10.,
                numeric_range_end=INFINITY + 1,
                categorical_indicators={"a", "b", "c"}
            )

    def test_range_rejects_invalid_numeric_range(self):

        with self.assertRaises(ValueError):
            Range(
                numeric_range_start=10.,
                numeric_range_end=1.,
                categorical_indicators={"a", "b", "c"}
            )

    def test_range_rejects_numbers_in_char_indicator_part(self):

        with self.assertRaises(TypeError):
            Range(
                numeric_range_start=10.,
                numeric_range_end=45.,
                categorical_indicators={"a", 0.34, "c"}
            )

    def test_range_does_not_reject_non_floats_in_numeric_part(self):
        Range(
            numeric_range_start=10,
            numeric_range_end=45,
            categorical_indicators={"a", "b", "c"}
        )

    def test_range_accepts_list(self):
        Range(
            numeric_range_start=10,
            numeric_range_end=45,
            categorical_indicators=["a", "b", "c"]
        )

    def test_feature_rejects_without_name(self):

        with self.assertRaises(TypeError):
            Feature(
                name=None,
                range=Range(
                    numeric_range_start=10,
                    numeric_range_end=45,
                    categorical_indicators={"a", "b", "c"}
                )
            )

    def test_feature_instantiates_with_name(self):
        Feature(
            name="feature",
            range=Range(
                numeric_range_start=10,
                numeric_range_end=45,
                categorical_indicators={"a", "b", "c"}
            )
        )

    def test_woe_bin_fails_without_value(self):

        with self.assertRaises(TypeError):
            WoEBin(
                bin_event_count=10,
                bin_non_event_count=100,
                woe=None,
                iv=0.4,
                bin=Range(
                    numeric_range_start=10,
                    numeric_range_end=45,
                    categorical_indicators={"a", "b", "c"}
                )
            )

    def test_woe_bin_fails_without_numeric_value(self):

        with self.assertRaises(ValueError):
            WoEBin(
                woe="abc",
                bin_event_count=10,
                bin_non_event_count=100,
                iv=0.4,
                bin=Range(
                    numeric_range_start=10,
                    numeric_range_end=45,
                    categorical_indicators={"a", "b", "c"}
                )
            )

    def test_woe_bin_succeeds_with_numeric_value(self):
        WoEBin(
            woe="12",
            bin_event_count=10,
            bin_non_event_count=100,
            iv=0.4,
            bin=Range(
                numeric_range_start=10,
                numeric_range_end=45,
                categorical_indicators={"a", "b", "c"}
            )
        )

    def test_woe_bin_fails_with_invalid_range(self):

        with self.assertRaises(ValueError):
            WoEBin(
                woe=12,
                bin_event_count=10,
                bin_non_event_count=100,
                iv=0.4,
                bin=Range(
                    numeric_range_start=10,
                    numeric_range_end=9,
                    categorical_indicators={"a", "b", "c"}
                )
            )
