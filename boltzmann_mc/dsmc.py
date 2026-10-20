"""Bird's Direct Simulation Monte Carlo for the space-homogeneous problem.

Implements Algorithm 4.1 (Maxwell molecules) and Algorithm 4.2 (general VHS
molecules, acceptance-rejection against a majorant) of the report.

The ensemble of Np simulation particles evolves as a continuous-time Markov
jump process: a collision clock ``tc`` is advanced after every (accepted)
collision, and collisions are performed until the clock reaches the end of
the macroscopic time step.
"""

import numpy as np

from .collisions import VHS, post_collision
from .sampling import uniform_sphere


class DSMC:
    """Space-homogeneous Bird scheme.

    Parameters
    ----------
    v : (Np, 3) array
        Initial velocities, sampled from the initial density f0.
    model : VHS
        Collision model. ``alpha == 0`` selects Algorithm 4.1, otherwise 4.2.
    eps : float
        Knudsen number epsilon.
    rho : float
        Number density (conserved in the collision step).
    rng : numpy.random.Generator
    """

    def __init__(self, v, model=None, eps=1.0, rho=1.0, rng=None):
        self.v = np.array(v, dtype=float)          # working copy
        self.Np = self.v.shape[0]
        self.model = model if model is not None else VHS()
        self.eps = float(eps)
        self.rho = float(rho)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.tc = 0.0          # collision clock
        self.t = 0.0           # end of the last macroscopic step
        self.stats = {"candidates": 0, "collisions": 0, "bound_updates": 0}

    # ------------------------------------------------------------------
    def advance(self, dt):
        """Advance the ensemble by one macroscopic time step ``dt``."""
        self.t += dt
        if self.model.is_maxwell:
            self._advance_maxwell()
        else:
            self._advance_vhs()

    # ------------------------------------------------------------------
    def _draw_pair(self):
        """Uniform unordered pair (i, j), i != j (Algorithm B.6 with n = 2)."""
        rng, N = self.rng, self.Np
        i = int(N * rng.random())
        j = int((N - 1) * rng.random())
        if j >= i:
            j += 1
        return i, j

    def _collide(self, i, j):
        v = self.v
        omega = uniform_sphere(self.rng)
        v[i], v[j] = post_collision(v[i], v[j], omega)

    # ------------------------------------------------------------------
    def _advance_maxwell(self):
        """Algorithm 4.1: constant cross section, no rejection needed.

        Every tick of the clock, a uniformly chosen pair collides, with
        dtc = 2 eps / (Np mu),  mu = rho * sigma_M.
        """
        mu = self.rho * self.model.sigma_T(0.0)
        dtc = 2.0 * self.eps / (self.Np * mu)
        while self.tc < self.t:
            i, j = self._draw_pair()
            self._collide(i, j)
            self.tc += dtc
            self.stats["candidates"] += 1
            self.stats["collisions"] += 1

    def _advance_vhs(self):
        """Algorithm 4.2: acceptance-rejection against sigma_T(2 dv).

        A candidate pair is accepted when  Sigma * xi < sigma_ij  and each
        accepted collision advances the clock by  2 eps / (Np rho sigma_ij).
        """
        v, rng, model = self.v, self.rng, self.model
        # Majorant (4.12): |v_i - v_j| <= 2 dv, and sigma_T is nondecreasing.
        dv = np.max(np.linalg.norm(v - v.mean(axis=0), axis=1))
        Sigma = model.sigma_T(2.0 * dv)
        if Sigma <= 0.0:
            raise ValueError("All particles share the same velocity: no collisions possible.")

        while self.tc < self.t:
            i, j = self._draw_pair()
            d = v[i] - v[j]
            sij = model.sigma_T(np.sqrt(d @ d))
            self.stats["candidates"] += 1
            if sij > Sigma:
                # Post-collisional velocities can exceed the initial bound
                # (see the remark after Algorithm 4.2): update it on the fly.
                Sigma = sij
                self.stats["bound_updates"] += 1
            if Sigma * rng.random() < sij:
                self._collide(i, j)
                self.tc += 2.0 * self.eps / (self.Np * self.rho * sij)
                self.stats["collisions"] += 1
