"""
This package implements the No-U-Turn Sampler (NUTS) algorithm 6 from the NUTS
paper (Hoffman & Gelman, 2011).

Content
-------

The package mainly contains:
  nuts6                     return samples using the NUTS
  numerical_grad            return numerical estimate of the local gradient
  emcee                     emcee NUTS sampler
"""

from .pinnuts import pinnuts
from .helpers import PinNutsSampler_fn_wrapper
try:
    from .emcee_nuts import NUTSSampler
except ImportError:
    pass
