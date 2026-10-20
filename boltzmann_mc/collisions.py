"""Binary collisions and the VHS cross section (Sections 2.1 and 4.1.1)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VHS:
    """Variable Hard Sphere model, sigma(omega, r) = C * r**alpha.

    alpha = 0 : Maxwell molecules
    alpha = 1 : hard spheres
    The default C = 1/(4 pi) gives the simple total cross section
    sigma_T(r) = r**alpha (eq. 4.9).
    """

    alpha: float = 1.0
    C: float = 1.0 / (4.0 * np.pi)

    def sigma_T(self, r):
        """Total cross section sigma_T(r) = 4 pi C r^alpha, eq. (4.9)."""
        if self.alpha == 0.0:
            return 4.0 * np.pi * self.C
        return 4.0 * np.pi * self.C * r**self.alpha

    @property
    def is_maxwell(self):
        return self.alpha == 0.0


def post_collision(vi, vj, omega):
    """Post-collisional velocities of an elastic collision, eq. (2.4).

    v'   = (v + v*)/2 + |v - v*| omega / 2
    v'*  = (v + v*)/2 - |v - v*| omega / 2

    Momentum and kinetic energy are conserved exactly.
    """
    d = vi - vj
    speed = np.sqrt(d @ d)
    cm = 0.5 * (vi + vj)
    half = 0.5 * speed * omega
    return cm + half, cm - half
