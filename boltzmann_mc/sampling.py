"""Random sampling tools (Appendix B of the report).

Everything is built from independent uniform random numbers xi ~ U[0, 1]
drawn from a numpy ``Generator``.
"""

import numpy as np


def box_muller(rng, n):
    """n independent standard Gaussians via Box-Muller, eq. (B.3).

    rho = sqrt(-2 ln xi1), theta = 2 pi xi2, x = rho cos(theta), y = rho sin(theta).
    """
    m = (n + 1) // 2
    rho = np.sqrt(-2.0 * np.log(1.0 - rng.random(m)))  # 1 - xi avoids log(0)
    theta = 2.0 * np.pi * rng.random(m)
    z = np.concatenate([rho * np.cos(theta), rho * np.sin(theta)])
    return z[:n]


def sample_maxwellian(rng, n, u=(0.0, 0.0, 0.0), T=1.0):
    """n velocities from the Maxwellian M(rho, u, T) of eq. (2.13).

    The three components are independent Gaussians with mean u_i and
    variance T (factorised case, Section B.2.1).
    """
    z = box_muller(rng, 3 * n).reshape(n, 3)
    return np.asarray(u, dtype=float) + np.sqrt(T) * z


def uniform_sphere(rng):
    """One unit vector uniformly distributed on S^2, eq. (B.4).

    phi = 2 pi xi1, theta = arccos(1 - 2 xi2).
    """
    phi = 2.0 * np.pi * rng.random()
    cos_t = 1.0 - 2.0 * rng.random()
    sin_t = np.sqrt(1.0 - cos_t * cos_t)
    return np.array([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t])


def sample_without_repetition(rng, N, n):
    """n distinct indices out of {0, ..., N-1}, uniformly (Algorithm B.6).

    Swap-to-the-end procedure: the chosen slot is overwritten with the last
    active one, so each draw costs O(1) and the pool stays uniform.
    """
    ind = list(range(N))
    M = N
    out = np.empty(n, dtype=np.int64)
    for k in range(n):
        j = int(M * rng.random())      # j uniform on {0, ..., M-1}
        out[k] = ind[j]
        ind[j] = ind[M - 1]
        M -= 1
    return out
