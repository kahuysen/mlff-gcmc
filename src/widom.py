"""Widom insertion of one CH4 in an empty, rigid framework: Henry coefficient at infinite dilution.

Frenkel & Smit 3rd ed. section 8.5.1. At low pressure n = K_H * p, with
K_H = <exp(-U/T)> / (R T rho_fw)  in mol/kg/Pa.
"""

import numpy as np

from energy import energy_fast

R = 8.31446261815324  # J/(mol K)
AVOGADRO = 6.02214076e23  # 1/mol
U_TO_KG = 1.66053906660e-27  # kg per atomic mass unit


def boltzmann(U, T):
    """exp(-U/T) with U in K."""
    return np.exp(-U / T)


def widom(N, T, positions, eps_pair, sigma_pair, L, rng, n_blocks=10):
    """N uniform insertions in the cell. Returns <exp(-U/T)> and its standard error from block averages."""
    assert N % n_blocks == 0, "N must be divisible by n_blocks"
    random_positions = rng.random((N, 3)) * L
    U = np.array([energy_fast(pos, positions, eps_pair, sigma_pair, L) for pos in random_positions])
    boltzmann_factors = boltzmann(U, T)
    block_means = boltzmann_factors.reshape(n_blocks, -1).mean(axis=1)
    sem = block_means.std(ddof=1) / np.sqrt(n_blocks)
    return boltzmann_factors.mean(), sem


def rho_framework(masses, L):
    """Mass density of the framework in kg/m^3. masses in u, L in A."""
    total_mass = np.sum(masses) * U_TO_KG
    volume = (L * 1e-10) ** 3
    return total_mass / volume


def henry_coefficient(boltzmann_mean, T, rho_fw):
    """K_H in mol/kg/Pa from <exp(-U/T)>, T in K and rho_fw in kg/m^3."""
    return boltzmann_mean / (rho_fw * R * T)
