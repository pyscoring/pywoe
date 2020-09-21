# -*- coding: utf-8 -*-
"""
Data models for serialising Weight-of-Evidence transformation information.
"""

import attr

from typing import FrozenSet
from pywoe.data_models.base import Range
from pywoe.data_models.feature import Feature
from pywoe.data_models.utils import check_validity_of_ranges


@attr.s(frozen=True)
class WoEBin(object):
    """
    A definition of the boundaries of a single bin.
    """

    woe: float = attr.ib(
        validator=attr.validators.instance_of(float),
        converter=float
    )
    """
    The Weight-of-Evidence value.
    """

    bin: Range = attr.ib(
        validator=attr.validators.instance_of(Range)
    )
    """
    The bin to which the WoE transformation will be applied to.
    """


@attr.s(frozen=True)
class WoESpec(object):
    """
    A template that defines binning for a feature.
    """

    feature: Feature = attr.ib(
        validator=attr.validators.instance_of(Feature)
    )
    """
    The feature to which WoE transformation is applied.
    """

    bins: FrozenSet[WoEBin] = attr.ib(
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(WoEBin),
            iterable_validator=attr.validators.instance_of(frozenset)
        ),
        converter=frozenset
    )
    """
    The specification of bins and WoE values applied to them.
    """

    def __attrs_post_init__(self) -> None:
        """
        A hook to check that the binning definition is well-defined.

        :raises: :any:`ValueError`
        """

        check_validity_of_ranges(
            feature=self.feature,
            bin_ranges=set([
                bin.bin for bin in self.bins
            ])
        )


