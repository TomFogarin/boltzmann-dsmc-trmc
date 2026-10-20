"""Relaxation of two counter-propagating beams to the Maxwellian equilibrium.

Space-homogeneous Boltzmann equation with Maxwell molecules (alpha = 0),
Knudsen number eps = 1 (kinetic regime), mu = rho * sigma_M = 1.

For Maxwell molecules the anisotropy
    D(t) = <c_x^2> - (<c_y^2> + <c_z^2>) / 2
obeys a closed equation, D(t) = D(0) exp(-mu t / (2 eps)).

One first-order TRMC step multiplies the expected anisotropy by
(1 - tau)(1 + tau/2), tau = 1 - exp(-mu dt / eps): a particle is untouched
(weight 1 - tau), or collides once (weight (1 - tau) tau, which halves the
anisotropy), or is thermalised (weight tau^2, anisotropy 0). This agrees with
the exact factor sqrt(1 - tau) only to first order in tau. We compare:
  - Bird's DSMC (Algorithm 4.1),
  - first-order TRMC (Algorithm 4.3) with two different time steps,
  - the analytical decay.

Usage:  python examples/01_relaxation_maxwell.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boltzmann_mc import DSMC, TRMC, VHS, diagnostics  # noqa: E402

NP = 4000          # simulation particles
RUNS = 8           # independent runs averaged
EPS = 1.0
T_END = 6.0
DT_OUT = 0.25      # DSMC output interval


def two_beams(rng, n, speed=1.5, spread=0.3):
    """Initial datum f0: two beams along x with small isotropic thermal noise."""
    v = spread * rng.standard_normal((n, 3))
    v[:, 0] += speed * rng.choice([-1.0, 1.0], size=n)
    return v


def run(cls, dt, seed):
    rng = np.random.default_rng(seed)
    v0 = two_beams(rng, NP)
    sim = cls(v0, VHS(alpha=0.0), eps=EPS, rng=rng)
    steps = int(round(T_END / dt))
    t = [0.0]
    D = [diagnostics.anisotropy(v0)]
    for _ in range(steps):
        sim.advance(dt)
        t.append(sim.t)
        D.append(diagnostics.anisotropy(sim.v))
    drift_P = np.abs(sim.v.sum(axis=0) - v0.sum(axis=0)).max()
    drift_E = abs(np.sum(sim.v**2) / np.sum(v0**2) - 1.0)
    return np.array(t), np.array(D), sim, v0, drift_P, drift_E


def average(cls, dt):
    out = [run(cls, dt, seed) for seed in range(RUNS)]
    t = out[0][0]
    D = np.mean([o[1] for o in out], axis=0)
    return t, D, out[0][2], out[0][3], max(o[4] for o in out), max(o[5] for o in out)


def main():
    results = {
        "DSMC (Alg. 4.1)": average(DSMC, DT_OUT),
        "TRMC, dt = 0.25": average(TRMC, 0.25),
        "TRMC, dt = 1.0": average(TRMC, 1.0),
    }

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: decay of the anisotropy
    t_ref = np.linspace(0, T_END, 200)
    D0 = results["DSMC (Alg. 4.1)"][1][0]
    for label, (t, D, *_rest) in results.items():
        ax[0].semilogy(t, np.abs(D) / D0, "o-", ms=3, label=label)
    ax[0].semilogy(t_ref, np.exp(-t_ref / (2 * EPS)), "k--", label=r"exact $e^{-\mu t/2\varepsilon}$")
    # One TRMC step multiplies D by (1 - tau)(1 + tau/2) (see docstring below).
    for dt, color in ((0.25, "C1"), (1.0, "C2")):
        tau = 1.0 - np.exp(-dt / EPS)
        factor = (1.0 - tau) * (1.0 + tau / 2.0)
        n = np.arange(0, int(round(T_END / dt)) + 1)
        ax[0].semilogy(n * dt, factor**n, ":", color=color, lw=1.2)
    ax[0].plot([], [], "k:", lw=1.2, label="TRMC prediction")
    ax[0].set_xlabel("time t")
    ax[0].set_ylabel(r"$D(t)/D(0)$")
    ax[0].set_title(f"Anisotropy decay (Maxwell molecules, $\\varepsilon={EPS}$)")
    ax[0].set_ylim(5e-3, 1.5)
    ax[0].legend(fontsize=8)

    # Right: histogram of v_x at t = 0, 6 and 15 (one DSMC run) vs Maxwellian
    rng = np.random.default_rng(0)
    v0 = two_beams(rng, NP)
    sim = DSMC(v0, VHS(alpha=0.0), eps=EPS, rng=rng)
    snaps = {0.0: v0.copy()}
    for t_snap in (6.0, 15.0):
        sim.advance(t_snap - sim.t)
        snaps[t_snap] = sim.v.copy()
    bins = np.linspace(-4, 4, 60)
    for t_snap, v in snaps.items():
        ax[1].hist(v[:, 0], bins=bins, density=True, alpha=0.45, label=f"DSMC, t = {t_snap:g}")
    x = np.linspace(-4, 4, 400)
    _, T, _ = diagnostics.moments(snaps[15.0])
    ax[1].plot(x, diagnostics.gaussian_pdf(x, 0.0, T), "k", lw=1.5, label="Maxwellian, same T")
    ax[1].set_xlabel(r"$v_x$")
    ax[1].set_ylabel("density")
    ax[1].set_title("Velocity distribution (higher moments relax more slowly)")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    os.makedirs("docs", exist_ok=True)
    fig.savefig("docs/relaxation_maxwell.png", dpi=150)

    print(f"Np = {NP}, runs = {RUNS}, eps = {EPS}, mu = 1")
    print("\nConservation (max over runs):")
    for label, (_, _, sim, _, dP, dE) in results.items():
        print(f"  {label:18s} max|drift of total momentum| = {dP:.2e}   rel. drift of total energy = {dE:.2e}")
    print("\nAnisotropy D(t)/D(0) at selected times:")
    print("  t      exact    " + "   ".join(f"{k:>16s}" for k in results))
    for tt in (1.0, 2.0, 3.0, 4.0):
        row = [f"{tt:4.1f}  {np.exp(-tt / (2 * EPS)):7.3f}"]
        for label, (t, D, *_r) in results.items():
            k = int(np.argmin(np.abs(t - tt)))
            row.append(f"{D[k] / D0:16.3f}")
        print("  " + "   ".join(row))
    print("\nExcess kurtosis of v_x (0 for a Maxwellian), single DSMC run:")
    for t_snap, v in snaps.items():
        print(f"  t = {t_snap:4.1f}: {diagnostics.excess_kurtosis_x(v):6.3f}")
    print("\nFigure saved to docs/relaxation_maxwell.png")


if __name__ == "__main__":
    main()
