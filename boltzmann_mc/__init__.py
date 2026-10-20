"""Toy Monte Carlo particle schemes for the Boltzmann equation.

Companion code to the report "Monte Carlo Methods for the Boltzmann Equation:
From DSMC to Time Relaxed Schemes" (after Pareschi and Russo).
"""

from .collisions import VHS, post_collision
from .dsmc import DSMC
from .trmc import TRMC, pullin_maxwellian
from .sampling import (box_muller, sample_maxwellian, sample_without_repetition,
                       uniform_sphere)
from . import diagnostics

__all__ = ["VHS", "post_collision", "DSMC", "TRMC", "pullin_maxwellian",
           "box_muller", "sample_maxwellian", "sample_without_repetition",
           "uniform_sphere", "diagnostics"]
