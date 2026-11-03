"""Lennard-Jones energy of a CH4 (united atom) in a rigid MOF.

Units: eps in K (energy / k_B), sigma and distances in A. Hence beta*U = U/T.
Shifted potential: u(r) - u(r_c) for r < r_c, 0 beyond. No tail correction.
"""

import numpy as np
import ase.io


def load_framework(cif_path):
    """Read a CIF and return (positions [N,3], symbols [N], L, masses [N] in u).

    Cubic cells only.
    """
    atoms = ase.io.read(cif_path, format="cif")
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    cell = atoms.cell.lengths()
    assert np.allclose(cell, cell[0]), f"cell is not cubic: {cell}"
    masses = atoms.get_masses()
    return positions, symbols, cell[0], masses


def load_force_field(ff_path, elements=("O_", "C_", "H_", "Zn_")):
    """Read (eps [K], sigma [A]) per element from a RASPA .def file.

    Keys without the underscore, so they match the symbols from the CIF.
    """
    lj_params = {}
    with open(ff_path) as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] in elements:
                lj_params[parts[0].rstrip("_")] = (float(parts[2]), float(parts[3]))
    missing = set(e.rstrip("_") for e in elements) - set(lj_params)
    assert not missing, f"not found in {ff_path}: {missing}"
    return lj_params


def lorentz_berthelot(p1, p2):
    """Mixing rules: eps_ij = sqrt(eps_i eps_j), sigma_ij = (sigma_i + sigma_j) / 2."""
    eps_1, sigma_1 = p1
    eps_2, sigma_2 = p2
    return ((eps_1 * eps_2) ** 0.5, (sigma_1 + sigma_2) / 2)


def pairwise_lj_params(symbols, lj_params, guest):
    """Mixed (eps, sigma) with the guest for every framework atom. Returns two arrays of length N."""
    pairs = np.array([lorentz_berthelot(guest, lj_params[s]) for s in symbols])
    assert pairs.shape == (len(symbols), 2)
    return pairs[:, 0], pairs[:, 1]


def distances_to_framework(positions, ch4_position, L):
    """Minimum-image distances from one point to all framework atoms in a cubic cell."""
    delta = positions - ch4_position
    delta -= np.round(delta / L) * L
    return np.linalg.norm(delta, axis=1)


def energy_fast(ch4_position, positions, eps_pair, sigma_pair, L, cutoff_distance=12.0):
    """Shifted LJ energy [K] of a CH4 at ch4_position against the framework."""
    assert cutoff_distance < L / 2, "cutoff must be smaller than half the cell edge"
    distances = distances_to_framework(positions, ch4_position, L)
    mask = distances < cutoff_distance
    d = distances[mask]
    eps = eps_pair[mask]
    sigma = sigma_pair[mask]
    energy = 4 * eps * ((sigma / d) ** 12 - (sigma / d) ** 6)
    energy_cutoff = 4 * eps * ((sigma / cutoff_distance) ** 12 - (sigma / cutoff_distance) ** 6)
    return energy.sum() - energy_cutoff.sum()
