"""Macroscopic quantities computed from a particle ensemble (eq. 2.14)."""

import numpy as np


def moments(v):
    """Mean velocity u, temperature T = <|v-u|^2>/3 and kinetic energy sum |v|^2."""
    u = v.mean(axis=0)
    T = np.mean(np.sum((v - u) ** 2, axis=1)) / 3.0
    return u, T, float(np.sum(v**2))


def anisotropy(v):
    """D = <c_x^2> - (<c_y^2> + <c_z^2>)/2 with c = v - u.

    D = 0 at equilibrium. For Maxwell molecules it decays exactly as
    D(t) = D(0) exp(-mu t / (2 eps)), with mu = rho * sigma_M.
    """
    c = v - v.mean(axis=0)
    s = np.mean(c**2, axis=0)
    return float(s[0] - 0.5 * (s[1] + s[2]))


def excess_kurtosis_x(v):
    """Excess kurtosis of v_x (zero for a Gaussian)."""
    c = v[:, 0] - v[:, 0].mean()
    return float(np.mean(c**4) / np.mean(c**2) ** 2 - 3.0)


def gaussian_pdf(x, mean, var):
    return np.exp(-((x - mean) ** 2) / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)
