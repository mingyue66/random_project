"""
Optimizer and LR scheduler utilities for Framework.

Exports a small, stable API:
- ScaledAdam
- LRScheduler, Eden, Eden2, Eden3
- get_parameter_groups_with_lrs
"""

from .optimizer import ScaledAdam
from .scheduler import Eden, Eden2, Eden3, LRScheduler
from .utils import get_parameter_groups_with_lrs

__all__ = [
    "ScaledAdam",
    "LRScheduler",
    "Eden",
    "Eden2",
    "Eden3",
    "get_parameter_groups_with_lrs",
]
