# -*- coding: utf-8 -*-
"""
Data models for binning specification. It's best to save binning in a tailor-made object to make it reusable
and possible to easily re-apply across different datasets.
"""

import attr

from typing import FrozenSet
from pywoe.data_models.base import Range
from pywoe.data_models.feature import Feature
from pywoe.data_models.utils import check_validity_of_ranges


@attr.s(frozen=True)
class BinningSpec(object):
    """
    A template that defines binning for a feature.
    """

    feature: Feature = attr.ib(
        validator=attr.validators.instance_of(Feature)
    )
    """
    The feature to which binning is applied.
    """

    bins: FrozenSet[Range] = attr.ib(
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(Range),
            iterable_validator=attr.validators.instance_of(frozenset)
        ),
        converter=frozenset
    )
    """
    The specification of bins.
    """

    def __attrs_post_init__(self) -> None:
        """
        A hook to check that the binning definition is well-defined.

        :raises: :any:`ValueError`
        """

        check_validity_of_ranges(
            feature=self.feature,
            bin_ranges=self.bins
        )
