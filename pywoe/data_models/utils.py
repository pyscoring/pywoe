# -*- coding: utf-8 -*-
"""
Utility methods used in definitions of data models.
"""

import itertools

from typing import FrozenSet, List
from collections import Counter
from pywoe.data_models.base import Range
from pywoe.data_models.feature import Feature
from pywoe.constants import NUMERIC_ACCURACY


def check_validity_of_ranges(
        feature: Feature,
        bin_ranges: FrozenSet[Range],
        numeric_accuracy: float = NUMERIC_ACCURACY
) -> None:
    """
    Checks if a defined set of ranges is applicable to a feature.

    :param feature: the feature in question
    :param bin_ranges: the set of ranges we want to check
    :param numeric_accuracy: the minimum difference of floats needed to deem them equal
    :raises: :any:`ValueError` in case the range is not valid
    """

    min_numeric = min([
        bin.numeric_range_start for bin in bin_ranges
    ])
    max_numeric = max([
        bin.numeric_range_end for bin in bin_ranges
    ])
    all_chars = list(
        itertools.chain.from_iterable(
            bin.categorical_indicators for bin in bin_ranges
        )
    )
    set_difference = feature.range.categorical_indicators - frozenset(all_chars)

    if min_numeric > feature.range.numeric_range_start:
        raise ValueError(
            "The minimum numeric value for feature `{name}` is beyond the binning spec ({value}).".format(
                name=feature.name,
                value=min_numeric
            )
        )

    elif max_numeric < feature.range.numeric_range_end:
        raise ValueError(
            "The maximum numeric value for feature `{name}` is beyond the binning spec ({value}).".format(
                name=feature.name,
                value=max_numeric
            )
        )

    elif len(set_difference) > 0:
        raise ValueError(
            "Some categorical indicators in feature `{name}` are missing from binning spec: {set}.".format(
                name=feature.name,
                set=list(set_difference)
            )
        )

    elif _categorical_value_intersection(all_chars):
        raise ValueError(
            "Some categorical indicators in feature `{name}` are specified in multiple bins.".format(
                name=feature.name
            )
        )

    elif _numeric_range_is_disjoint(bin_ranges, numeric_accuracy):
        raise ValueError(
            "The numeric ranges of bins either contains gaps or overlaps."
        )


def _numeric_range_is_disjoint(
        bin_ranges: FrozenSet[Range],
        numeric_accuracy: float
) -> bool:
    """
    A method that checks if the numeric range bins are disjoint and don't contain gaps.

    :param bin_ranges: the set of ranges we want to check
    :param numeric_accuracy: the minimum difference of floats needed to deem them equal
    :return: `True` if the range is disjoint, `False` otherwise
    """

    bin_starts = sorted([bin.numeric_range_start for bin in bin_ranges])
    bin_ends = sorted([bin.numeric_range_end for bin in bin_ranges])

    if len(bin_starts) <= 1:
        return False

    else:

        for i, bin_start in enumerate(bin_starts[1:]):

            if abs(bin_ends[i] - bin_start) > numeric_accuracy:
                return True

        return False


def _categorical_value_intersection(
        all_chars: List
) -> bool:
    """
    A method that finds if there are items that appear more than once in a list

    :param all_chars: the list we want to check
    :return: `True` if there is an intersection, `False` otherwise
    """

    maximum_count = max(Counter(all_chars).values())
    return maximum_count > 1
