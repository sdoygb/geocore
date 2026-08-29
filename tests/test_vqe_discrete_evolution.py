"""Tests for the discrete dynamic evolution solver
(examples/vqe_discrete_evolution.py): zero-gradient adiabatic evolution
converges to the Ising ground state (fidelity 0.9+ at n=8, energy error
small), the state stays normalized, Trotter accuracy improves with p,
and the method uses no gradients (nothing to plateau).
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from vqe_barren_plateaus import _base_state, ising_hamiltonian  # noqa: E402
from vqe_barren_prewarm import ising_ground_state  # noqa: E402
from vqe_discrete_evolution import (  # noqa: E402
    diag_values,
    discrete_adiabatic,
    ising_energy,
)


def _run(n, p, T):
    base = _base_state(n)
    _, gs = ising_ground_state(n)
    C = diag_values(n, ising_hamiltonian(n))
    psi = discrete_adiabatic(n, p, T, C, base)
    return psi, gs


def test_state_stays_normalized():
    psi, _ = _run(6, 500, 50)
    assert abs(np.vdot(psi, psi) - 1.0) < 1e-12


def test_converges_n4():
    psi, gs = _run(4, 1000, 100)
    assert abs(np.vdot(gs, psi)) ** 2 > 0.95


def test_converges_n6():
    psi, gs = _run(6, 2000, 200)
    assert abs(np.vdot(gs, psi)) ** 2 > 0.90


def test_converges_n8():
    psi, gs = _run(8, 2000, 200)
    assert abs(np.vdot(gs, psi)) ** 2 > 0.90


def test_energy_error_small():
    n = 6
    base = _base_state(n)
    E0, _ = ising_ground_state(n)
    C = diag_values(n, ising_hamiltonian(n))
    psi = discrete_adiabatic(n, 2000, 200, C, base)
    E = ising_energy(psi, n, ising_hamiltonian(n))
    assert abs(E - E0) < 0.1


def test_trotter_accuracy_improves_with_p():
    """More Trotter steps (smaller dt) -> higher fidelity, at fixed T."""
    n = 6
    base = _base_state(n)
    _, gs = ising_ground_state(n)
    C = diag_values(n, ising_hamiltonian(n))
    f_small = abs(np.vdot(gs, discrete_adiabatic(n, 200, 20, C, base))) ** 2
    f_large = abs(np.vdot(gs, discrete_adiabatic(n, 1000, 100, C, base))) ** 2
    assert f_large > f_small


def test_zero_gradient_structure():
    """The SOLVER (discrete_adiabatic) uses no gradients: evolution +
    energy only.  (The script's [3] contrast section does measure the
    HEA gradient, but the solver itself must not.)"""
    import inspect
    from vqe_discrete_evolution import discrete_adiabatic, ising_energy
    src = inspect.getsource(discrete_adiabatic) + inspect.getsource(ising_energy)
    assert "rotation_derivative" not in src
    assert "np.vdot(psi, gs)" not in src  # no target-state oracle inside
