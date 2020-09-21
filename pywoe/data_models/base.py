# -*- coding: utf-8 -*-
"""
Basic classes used as simple building blocks elsewhere.
"""

import attr

from typing import FrozenSet


@attr.s(frozen=True)
class Range(object):
    """
    A definition of the boundaries of a single range. Used to define a bin, a feature range.
    We make an assumption that the numeric range is contiguous, i.e. there are no disconnected
    numeric range parts.
    """

    numeric_range_start: float = attr.ib(
        validator=attr.validators.instance_of(float),
        converter=float
    )
    """
    The place where numeric range starts.
    """

    numeric_range_end: float = attr.ib(
        validator=attr.validators.instance_of(float),
        converter=float
    )
    """
    The place where numeric range ends.
    """

    categorical_indicators: FrozenSet[str] = attr.ib(
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(str),
            iterable_validator=attr.validators.instance_of(frozenset)
        ),
        converter=frozenset
    )
    """
    The set of categorical/char values in the range.
    """

    def __attrs_post_init__(self) -> None:
        """
        A hook to check that the range is well-defined.
        """

        if self.numeric_range_start > self.numeric_range_end:
            raise ValueError("Numeric range ends before it starts: start = {start}, end = {end}".format(
                start=self.numeric_range_start,
                end=self.numeric_range_end
            ))
