# -*- coding: utf-8 -*-
"""
Feature specification, allowing us to serialise information we learn about a feature
(e.g. numeric data range, possible categorical/char values) during training.
"""

import attr

from pywoe.data_models.base import Range


@attr.s(frozen=True)
class Feature(object):
    """
    A definition of the boundaries of a single feature.
    """

    name: str = attr.ib(
        validator=attr.validators.instance_of(str)
    )
    """
    The name of the feature.
    """

    range: Range = attr.ib(
        validator=attr.validators.instance_of(Range)
    )
    """
    The range of the feature.
    """
