"""Widom-insertie van een CH4 in een leeg, rigide rooster: Henry-coefficient bij oneindige verdunning.

Frenkel & Smit 3e ed. par. 8.5.1. Bij lage druk: n = K_H * p, met
K_H = <exp(-U/T)> / (R T rho_fw)  in mol/kg/Pa.
"""

import numpy as np

from energy import energy_fast

R = 8.31446261815324  # J/(mol K)
AVOGADRO = 6.02214076e23  # 1/mol
U_TO_KG = 1.66053906660e-27  # kg per atomaire massa-eenheid


def boltzmann(U, T):
    """exp(-U/T) met U in K."""
    return np.exp(-U / T)


def widom(N, T, positions, eps_pair, sigma_pair, L, rng, n_blocks=10):
    """N uniforme inserties in de cel. Geeft <exp(-U/T)> en de standaardfout via blokgemiddelden."""
    assert N % n_blocks == 0, "N moet deelbaar zijn door n_blocks"
    random_positions = rng.random((N, 3)) * L
    U = np.array([energy_fast(pos, positions, eps_pair, sigma_pair, L) for pos in random_positions])
    boltzmann_factors = boltzmann(U, T)
    block_means = boltzmann_factors.reshape(n_blocks, -1).mean(axis=1)
    sem = block_means.std(ddof=1) / np.sqrt(n_blocks)
    return boltzmann_factors.mean(), sem


def rho_framework(masses, L):
    """Massadichtheid van het rooster in kg/m^3. masses in u, L in A."""
    total_mass = np.sum(masses) * U_TO_KG
    volume = (L * 1e-10) ** 3
    return total_mass / volume


def henry_coefficient(boltzmann_mean, T, rho_fw):
    """K_H in mol/kg/Pa uit <exp(-U/T)>, T in K en rho_fw in kg/m^3."""
    return boltzmann_mean / (rho_fw * R * T)
