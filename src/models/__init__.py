"""
Model definitions for the workshop paper.

Four downscaling models plus the learned forward operator A_theta.
"""

from .diffusion import DiffusionWrapper
from .geounet import GeoUNetWrapper
from .geodiff import GeoDiffWrapper
from .a_operator_CNN import AOperatorWrapper, load_a_operator
from .CorrDiff import CorrDiffWrapper

__all__ = [
    'DiffusionWrapper',
    'GeoUNetWrapper',
    'GeoDiffWrapper',
    'AOperatorWrapper',
    'load_a_operator',
    'CorrDiffWrapper',
]
