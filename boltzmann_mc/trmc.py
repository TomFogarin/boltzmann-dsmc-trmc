"""First-order Time Relaxed Monte Carlo (Algorithm 4.3 of the report).

Truncating the Wild sum at m = 1 gives

    f^{n+1} = A0 f^n + A1 f_1 + A2 M,
    A0 = 1 - tau,  A1 = (1 - tau) tau,  A2 = tau^2,
    tau = 1 - exp(-mu dt / eps),  mu = rho * Sigma.

At the particle level, A1 Np / 2 pairs undergo one candidate collision, A2 Np
particles are replaced by a sample of the local Maxwellian, and the rest is
left untouched. When mu dt / eps -> infinity, tau -> 1 and the whole ensemble
is replaced by a Maxwellian: the scheme is asymptotic preserving.
"""

import numpy as np

from .collisions import VHS, post_collision
from .sampling import box_muller, sample_without_repetition, uniform_sphere


def pullin_maxwellian(rng, P, E, n):
    """n velocities with *exactly* total momentum P and sum |v|^2 = E.

    Pullin's matching idea: draw Gaussians, remove their sample mean, rescale
    them to the target thermal energy, then add the target mean velocity.
    The result is Maxwellian-like but conserves the empirical moments of the
    particles it replaces, which prevents statistical drift of the ensemble.
    """
    u = P / n
    thermal = max(E - n * float(u @ u), 0.0)
    z = box_muller(rng, 3 * n).reshape(n, 3)
    z -= z.mean(axis=0)
    z *= np.sqrt(thermal / np.sum(z * z))
    return u + z


class TRMC:
    """Space-homogeneous first-order TRMC scheme (same interface as ``DSMC``)."""

    def __init__(self, v, model=None, eps=1.0, rho=1.0, rng=None):
        self.v = np.array(v, dtype=float)
        self.Np = self.v.shape[0]
        self.model = model if model is not None else VHS()
        self.eps = float(eps)
        self.rho = float(rho)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.t = 0.0
        self.stats = {"candidates": 0, "collisions": 0, "thermalized": 0}

    def advance(self, dt):
        """One TRMC step of length ``dt`` (any value of mu*dt/eps)."""
        v, rng, model, Np = self.v, self.rng, self.model, self.Np
        self.t += dt

        # Majorant Sigma >= sigma_ij as in Bird's scheme, and mu = rho * Sigma.
        dv = np.max(np.linalg.norm(v - v.mean(axis=0), axis=1))
        Sigma = model.sigma_T(2.0 * dv)
        mu = self.rho * Sigma

        tau = -np.expm1(-mu * dt / self.eps)          # 1 - exp(-mu dt / eps)
        A1 = (1.0 - tau) * tau
        A2 = tau * tau

        Nc = int(round(Np * A1 / 2.0))                # number of colliding pairs
        NM = int(round(Np * A2))                      # particles sent to the Maxwellian
        NM = min(NM, Np - 2 * Nc)                     # guard against rounding

        # Disjoint random subsets (Algorithm B.6): pairs first, then Maxwellian.
        idx = sample_without_repetition(rng, Np, 2 * Nc + NM)

        # Candidate collisions with majorized acceptance-rejection.
        for k in range(Nc):
            i, j = idx[2 * k], idx[2 * k + 1]
            d = v[i] - v[j]
            sij = model.sigma_T(np.sqrt(d @ d))
            self.stats["candidates"] += 1
            if Sigma * rng.random() < sij:
                v[i], v[j] = post_collision(v[i], v[j], uniform_sphere(rng))
                self.stats["collisions"] += 1

        # Maxwellian thermalisation that preserves the subset's P and E.
        if NM >= 2:
            sel = idx[2 * Nc: 2 * Nc + NM]
            sub = v[sel]
            v[sel] = pullin_maxwellian(rng, sub.sum(axis=0), float(np.sum(sub * sub)), NM)
            self.stats["thermalized"] += NM
