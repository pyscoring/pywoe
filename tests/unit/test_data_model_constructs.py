# -*- coding: utf-8 -*-
"""
Data model more complex construct tests. Mainly checks that validation works as expected.
"""

import unittest

from pywoe.data_models.base import Range
from pywoe.data_models.feature import Feature
from pywoe.data_models.binning import BinningSpec
from pywoe.data_models.woe import WoESpec, WoEBin


class DataModelConstructTests(unittest.TestCase):

    def test_invalid_range_not_accepted_binning(self):

        with self.assertRaises(ValueError):
            BinningSpec(
                feature=Feature(
                    name="feature",
                    range=Range(
                        numeric_range_start=0.5,
                        numeric_range_end=2.5,
                        categorical_indicators={"M", "C", "_"}
                    )
                ),
                bins={
                    Range(
                        numeric_range_start=0.2,
                        numeric_range_end=0.8,
                        categorical_indicators={"M", "C", "_"}
                    )
                }
            )

    def test_invalid_categorical_range_not_accepted_binning(self):

        with self.assertRaises(ValueError):
            BinningSpec(
                feature=Feature(
                    name="feature",
                    range=Range(
                        numeric_range_start=0.5,
                        numeric_range_end=2.5,
                        categorical_indicators={"M", "C", "_", "X"}
                    )
                ),
                bins={
                    Range(
                        numeric_range_start=0.5,
                        numeric_range_end=2.5,
                        categorical_indicators={"M", "C", "_"}
                    )
                }
            )

    def test_disjoint_range_not_accepted(self):

        with self.assertRaises(ValueError):
            BinningSpec(
                feature=Feature(
                    name="feature",
                    range=Range(
                        numeric_range_start=0.5,
                        numeric_range_end=2.5,
                        categorical_indicators={"M", "C", "_"}
                    )
                ),
                bins={
                    Range(
                        numeric_range_start=0.5,
                        numeric_range_end=1.4,
                        categorical_indicators={"M", "C"}
                    ),
                    Range(
                        numeric_range_start=1.5,
                        numeric_range_end=2.5,
                        categorical_indicators={"_"}
                    )
                }
            )


    def test_overlapping_range_not_accepted(self):

        with self.assertRaises(ValueError):
            BinningSpec(
                feature=Feature(
                    name="feature",
                    range=Range(
                        numeric_range_start=0.5,
                        numeric_range_end=2.5,
                        categorical_indicators={"M", "C", "_"}
                    )
                ),
                bins={
                    Range(
                        numeric_range_start=0.5,
                        numeric_range_end=1.7,
                        categorical_indicators={"M", "C"}
                    ),
                    Range(
                        numeric_range_start=1.5,
                        numeric_range_end=2.5,
                        categorical_indicators={"_"}
                    )
                }
            )

    def test_overlapping_categorical_not_accepted(self):

        with self.assertRaises(ValueError):
            BinningSpec(
                feature=Feature(
                    name="feature",
                    range=Range(
                        numeric_range_start=0.5,
                        numeric_range_end=2.5,
                        categorical_indicators={"M", "C", "_"}
                    )
                ),
                bins={
                    Range(
                        numeric_range_start=0.5,
                        numeric_range_end=1.5,
                        categorical_indicators={"M", "C"}
                    ),
                    Range(
                        numeric_range_start=1.5,
                        numeric_range_end=2.5,
                        categorical_indicators={"_", "C"}
                    )
                }
            )

    def test_valid_range_accepted_binning(self):
        BinningSpec(
            feature=Feature(
                name="feature",
                range=Range(
                    numeric_range_start=0.5,
                    numeric_range_end=2.5,
                    categorical_indicators={"M", "C", "_"}
                )
            ),
            bins={
                Range(
                    numeric_range_start=0.4,
                    numeric_range_end=2.6,
                    categorical_indicators={"M", "C", "_"}
                )
            }
        )

    def test_valid_composite_range_accepted_binning(self):
        BinningSpec(
            feature=Feature(
                name="feature",
                range=Range(
                    numeric_range_start=0.5,
                    numeric_range_end=2.5,
                    categorical_indicators={"M", "C", "_"}
                )
            ),
            bins={
                Range(
                    numeric_range_start=0.4,
                    numeric_range_end=1.5,
                    categorical_indicators={"_"}
                ),
                Range(
                    numeric_range_start=1.5,
                    numeric_range_end=2.6,
                    categorical_indicators={"M", "C"}
                )
            }
        )

    def test_woespec_with_valid_ranges_accepted(self):
        WoESpec(
            feature=Feature(
                name="feature",
                range=Range(
                    numeric_range_start=0.5,
                    numeric_range_end=2.5,
                    categorical_indicators={"M", "C", "_"}
                )
            ),
            bins={
                WoEBin(
                    woe=1.3,
                    bin_event_count=10,
                    bin_non_event_count=100,
                    iv=0.4,
                    bin=Range(
                        numeric_range_start=0.4,
                        numeric_range_end=1.5,
                        categorical_indicators={"_"}
                    )
                ),
                WoEBin(
                    woe=3.4,
                    bin_event_count=10,
                    bin_non_event_count=100,
                    iv=0.4,
                    bin=Range(
                        numeric_range_start=1.5,
                        numeric_range_end=2.6,
                        categorical_indicators={"M", "C"}
                    )
                )
            }
        )

    def test_woespec_with_invalid_ranges_not_accepted(self):

        with self.assertRaises(ValueError):
            WoESpec(
                feature=Feature(
                    name="feature",
                    range=Range(
                        numeric_range_start=0.5,
                        numeric_range_end=2.5,
                        categorical_indicators={"M", "C", "_"}
                    )
                ),
                bins={
                    WoEBin(
                        woe=1.3,
                        bin_event_count=10,
                        bin_non_event_count=100,
                        iv=0.4,
                        bin=Range(
                            numeric_range_start=0.4,
                            numeric_range_end=1.8,
                            categorical_indicators={"_"}
                        )
                    ),
                    WoEBin(
                        woe=3.4,
                        bin_event_count=10,
                        bin_non_event_count=100,
                        iv=0.4,
                        bin=Range(
                            numeric_range_start=1.5,
                            numeric_range_end=2.6,
                            categorical_indicators={"M", "C"}
                        )
                    )
                }
            )
