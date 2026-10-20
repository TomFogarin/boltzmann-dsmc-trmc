"""Sanity tests. Run with `pytest` or simply `python tests/test_core.py`."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from boltzmann_mc import (DSMC, TRMC, VHS, diagnostics, post_collision,  # noqa: E402
                          pullin_maxwellian, sample_maxwellian,
                          sample_without_repetition, uniform_sphere)


def two_beams(rng, n=400):
    sign = rng.choice([-1.0, 1.0], size=n)
    v = 0.3 * rng.standard_normal((n, 3))
    v[:, 0] += 1.5 * sign
    return v


def test_collision_conserves_momentum_and_energy():
    rng = np.random.default_rng(1)
    for _ in range(200):
        vi, vj = rng.normal(size=3), rng.normal(size=3)
        a, b = post_collision(vi, vj, uniform_sphere(rng))
        assert np.allclose(a + b, vi + vj)
        assert np.isclose(a @ a + b @ b, vi @ vi + vj @ vj)


def test_uniform_sphere_moments():
    rng = np.random.default_rng(2)
    w = np.array([uniform_sphere(rng) for _ in range(20000)])
    assert np.allclose(np.linalg.norm(w, axis=1), 1.0)
    assert np.all(np.abs(w.mean(axis=0)) < 0.02)
    assert np.all(np.abs((w**2).mean(axis=0) - 1 / 3) < 0.01)


def test_maxwellian_sampling():
    rng = np.random.default_rng(3)
    v = sample_maxwellian(rng, 50000, u=(1.0, 0.0, -1.0), T=2.0)
    u, T, _ = diagnostics.moments(v)
    assert np.allclose(u, [1.0, 0.0, -1.0], atol=0.03)
    assert abs(T - 2.0) < 0.05


def test_sampling_without_repetition():
    rng = np.random.default_rng(4)
    idx = sample_without_repetition(rng, 100, 100)
    assert sorted(idx.tolist()) == list(range(100))
    idx = sample_without_repetition(rng, 100, 30)
    assert len(set(idx.tolist())) == 30


def test_pullin_matches_moments_exactly():
    rng = np.random.default_rng(5)
    P, E, n = np.array([3.0, -1.0, 2.0]), 40.0, 25
    w = pullin_maxwellian(rng, P, E, n)
    assert np.allclose(w.sum(axis=0), P)
    assert np.isclose(np.sum(w * w), E)


def test_vhs_cross_section():
    assert np.isclose(VHS(alpha=0.0).sigma_T(3.0), 1.0)
    assert np.isclose(VHS(alpha=1.0).sigma_T(3.0), 3.0)


def _conservation(cls, model, eps, dt):
    rng = np.random.default_rng(6)
    v0 = two_beams(rng)
    sim = cls(v0, model, eps=eps, rng=rng)
    for _ in range(5):
        sim.advance(dt)
    assert np.allclose(sim.v.sum(axis=0), v0.sum(axis=0), atol=1e-9)
    assert np.isclose(np.sum(sim.v**2), np.sum(v0**2), rtol=1e-12)


def test_dsmc_maxwell_conservation():
    _conservation(DSMC, VHS(alpha=0.0), eps=1.0, dt=0.2)


def test_dsmc_hard_sphere_conservation():
    _conservation(DSMC, VHS(alpha=1.0), eps=1.0, dt=0.2)


def test_trmc_conservation_kinetic_and_fluid():
    _conservation(TRMC, VHS(alpha=1.0), eps=1.0, dt=0.2)
    _conservation(TRMC, VHS(alpha=1.0), eps=1e-4, dt=0.2)


def test_trmc_asymptotic_preserving():
    """For mu dt / eps >> 1 one step gives a Maxwellian.

    Pullin's step fixes P and E exactly, but the anisotropy of a finite
    Gaussian sample is only zero up to O(1/sqrt(N)) statistical noise.
    """
    rng = np.random.default_rng(7)
    N = 2000
    v0 = two_beams(rng, N)
    D0 = diagnostics.anisotropy(v0)
    sim = TRMC(v0, VHS(alpha=0.0), eps=1e-4, rng=rng)
    sim.advance(0.2)
    assert abs(diagnostics.anisotropy(sim.v)) < 4.0 / np.sqrt(N)
    assert abs(diagnostics.anisotropy(sim.v)) < 0.1 * D0
    assert abs(diagnostics.excess_kurtosis_x(sim.v)) < 0.3


def test_dsmc_maxwell_relaxation_rate():
    """D(t) = D(0) exp(-mu t / (2 eps)) for Maxwell molecules (mu = 1)."""
    rng = np.random.default_rng(8)
    v0 = two_beams(rng, 3000)
    sim = DSMC(v0, VHS(alpha=0.0), eps=1.0, rng=rng)
    D0 = diagnostics.anisotropy(v0)
    sim.advance(2.0)
    ratio = diagnostics.anisotropy(sim.v) / D0
    assert abs(ratio - np.exp(-1.0)) < 0.06


if __name__ == "__main__":
    tests = [(k, f) for k, f in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print("PASS", name)
    print(f"{len(tests)} tests passed")
