"""Cost of one macroscopic time step as the Knudsen number goes to zero.

Hard-sphere gas (alpha = 1) starting from two counter-propagating beams, one
macroscopic step of length dt = 0.1. The mean collision time of Bird's scheme
is proportional to eps, so the number of collisions per step grows like 1/eps
(Section 4.1.3), whereas first-order TRMC always processes at most Np/2
candidate pairs plus one Maxwellian resampling (Section 4.3.1).

For every eps we also report the anisotropy D after the step relative to its
initial value: for small eps both schemes should be (almost) at equilibrium.

Usage:  python examples/02_cost_vs_knudsen.py
"""

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boltzmann_mc import DSMC, TRMC, VHS, diagnostics  # noqa: E402

NP = 1000
DT = 0.1
EPS_LIST = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]


def two_beams(rng, n, speed=1.5, spread=0.3):
    v = spread * rng.standard_normal((n, 3))
    v[:, 0] += speed * rng.choice([-1.0, 1.0], size=n)
    return v


def one_step(cls, eps, seed=0):
    rng = np.random.default_rng(seed)
    v0 = two_beams(rng, NP)
    sim = cls(v0, VHS(alpha=1.0), eps=eps, rng=rng)
    t0 = time.perf_counter()
    sim.advance(DT)
    wall = time.perf_counter() - t0
    D_ratio = diagnostics.anisotropy(sim.v) / diagnostics.anisotropy(v0)
    return sim.stats, wall, D_ratio


def main():
    rows = []
    print(f"Hard spheres, Np = {NP}, dt = {DT}, one macroscopic step")
    header = (f"{'eps':>7} | {'DSMC coll.':>11} {'cand.':>9} {'time[s]':>8} {'D/D0':>7} "
              f"| {'TRMC cand.':>10} {'thermal.':>9} {'time[s]':>8} {'D/D0':>7}")
    print(header)
    print("-" * len(header))
    for eps in EPS_LIST:
        sd, wd, Dd = one_step(DSMC, eps)
        st, wt, Dt = one_step(TRMC, eps)
        rows.append((eps, sd["collisions"], sd["candidates"], wd, Dd,
                     st["candidates"], st["thermalized"], wt, Dt))
        print(f"{eps:7.3f} | {sd['collisions']:11d} {sd['candidates']:9d} {wd:8.3f} {Dd:7.3f} "
              f"| {st['candidates']:10d} {st['thermalized']:9d} {wt:8.3f} {Dt:7.3f}")

    r = np.array(rows)
    eps = r[:, 0]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(eps, r[:, 2], "o-", label="DSMC (candidate pairs)")
    ax[0].loglog(eps, r[:, 1], "s--", label="DSMC (accepted collisions)")
    ax[0].loglog(eps, r[:, 5] + r[:, 6], "^-", label="TRMC (candidate pairs + thermalised particles)")
    ax[0].axhline(NP, color="gray", lw=0.8, ls="--")
    ax[0].text(eps[-1], NP * 1.15, r"$N_p$", color="gray", fontsize=8)
    ax[0].loglog(eps, r[:, 0] ** -1.0 * r[0, 2] * eps[0], "k:", lw=1, label=r"slope $\propto 1/\varepsilon$")
    ax[0].set_xlabel(r"Knudsen number $\varepsilon$")
    ax[0].set_ylabel("operations per macroscopic step")
    ax[0].set_title(f"Collisions per step ($N_p={NP}$, $\\Delta t={DT}$)")
    ax[0].legend(fontsize=8)

    ax[1].loglog(eps, r[:, 3], "o-", label="DSMC")
    ax[1].loglog(eps, r[:, 7], "^-", label="TRMC")
    ax[1].set_xlabel(r"Knudsen number $\varepsilon$")
    ax[1].set_ylabel("wall-clock time per step [s]")
    ax[1].set_title("Run time (pure Python, toy implementation)")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    os.makedirs("docs", exist_ok=True)
    fig.savefig("docs/cost_vs_knudsen.png", dpi=150)
    print("\nFigure saved to docs/cost_vs_knudsen.png")


if __name__ == "__main__":
    main()
