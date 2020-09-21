# -*- coding: utf-8 -*-
"""
Basic classes used as simple building blocks elsewhere.
"""

import attr

from typing import FrozenSet
from pywoe.constants import INFINITY


def _x_smaller_than_infinity(instance, attribute, value):
    """
    Custom validator.

    :param instance: the `attrs` class instance
    :param attribute: the attribute
    :param value: the attribute value
    """

    if value is not None:

        if value >= INFINITY:
            raise ValueError("Each numeric value has to be smaller than infinity (={})".format(INFINITY))


@attr.s(frozen=True)
class Range(object):
    """
    A definition of the boundaries of a single range. Used to define a bin, a feature range.
    We make an assumption that the numeric range is contiguous, i.e. there are no disconnected
    numeric range parts.
    """

    numeric_range_start: float = attr.ib(
        validator=[
            attr.validators.instance_of((float, type(None))),
            _x_smaller_than_infinity
        ],
        converter=lambda x: float(x) if x is not None else x,
        default=None
    )
    """
    The place where numeric range starts.
    """

    numeric_range_end: float = attr.ib(
        validator=[
            attr.validators.instance_of((float, type(None))),
            _x_smaller_than_infinity
        ],
        converter=lambda x: float(x) if x is not None else x,
        default=None
    )
    """
    The place where numeric range ends.
    """

    categorical_indicators: FrozenSet[str] = attr.ib(
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(str),
            iterable_validator=attr.validators.instance_of(frozenset)
        ),
        converter=frozenset,
        default=frozenset()
    )
    """
    The set of categorical/char values in the range.
    """

    def __attrs_post_init__(self) -> None:
        """
        A hook to check that the range is well-defined.
        """

        if self.numeric_range_start is not None and self.numeric_range_end is not None:

            if self.numeric_range_start > self.numeric_range_end:
                raise ValueError("Numeric range ends before it starts: start = {start}, end = {end}".format(
                    start=self.numeric_range_start,
                    end=self.numeric_range_end
                ))

        elif self.numeric_range_start is not None and self.numeric_range_end is None:
            raise ValueError("Both numeric range start and end have to be specified!")

        elif self.numeric_range_start is None and self.numeric_range_end is not None:
            raise ValueError("Both numeric range start and end have to be specified!")
