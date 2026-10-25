"""Lennard-Jones energie van een CH4 (united atom) in een rigide MOF.

Eenheden: eps in K (energie / k_B), sigma en afstanden in A. Dus beta*U = U/T.
Shifted potential: u(r) - u(r_c) voor r < r_c, 0 daarbuiten. Geen tail correction.
"""

import numpy as np
import ase.io


def load_framework(cif_path):
    """Lees een CIF en geef (positions [N,3], symbols [N], L) terug. Alleen kubische cellen."""
    atoms = ase.io.read(cif_path, format="cif")
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    cell = atoms.cell.lengths()
    assert np.allclose(cell, cell[0]), f"cel is niet kubisch: {cell}"
    masses_dict = {"Zn": 65.38, "C": 12, "O":15.999, "H":1.008 }
    masses = np.array([masses_dict[symbol] for symbol in symbols])
    return positions, symbols, cell[0], masses, atoms


def load_force_field(ff_path, elements=("O_", "C_", "H_", "Zn_")):
    """Lees (eps [K], sigma [A]) per element uit een RASPA .def-bestand.

    Sleutels zonder underscore, zodat ze overeenkomen met de symbolen uit de CIF.
    """
    lj_params = {}
    with open(ff_path) as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] in elements:
                lj_params[parts[0].rstrip("_")] = (float(parts[2]), float(parts[3]))
    missing = set(e.rstrip("_") for e in elements) - set(lj_params)
    assert not missing, f"niet gevonden in {ff_path}: {missing}"
    return lj_params


def lorentz_berthelot(p1, p2):
    """Mengregels: eps_ij = sqrt(eps_i eps_j), sigma_ij = (sigma_i + sigma_j) / 2."""
    eps_1, sigma_1 = p1
    eps_2, sigma_2 = p2
    return ((eps_1 * eps_2) ** 0.5, (sigma_1 + sigma_2) / 2)


def pairwise_lj_params(symbols, lj_params, guest):
    """Per roosteratoom de gemengde (eps, sigma) met de gast. Geeft twee arrays van lengte N."""
    pairs = np.array([lorentz_berthelot(guest, lj_params[s]) for s in symbols])
    assert pairs.shape == (len(symbols), 2)
    return pairs[:, 0], pairs[:, 1]


def distances_to_framework(positions, ch4_position, L):
    """Minimum-image-afstanden van een punt tot alle roosteratomen in een kubische cel."""
    delta = positions - ch4_position
    delta -= np.round(delta / L) * L
    return np.linalg.norm(delta, axis=1)


def energy_fast(ch4_position, positions, eps_pair, sigma_pair, L, cutoff_distance=12.0):
    """Shifted LJ-energie [K] van een CH4 op ch4_position tegenover het rooster."""
    assert cutoff_distance < L / 2, "cutoff moet kleiner zijn dan de halve celribbe"
    distances = distances_to_framework(positions, ch4_position, L)
    mask = distances < cutoff_distance
    d = distances[mask]
    eps = eps_pair[mask]
    sigma = sigma_pair[mask]
    energy = 4 * eps * ((sigma / d) ** 12 - (sigma / d) ** 6)
    energy_cutoff = 4 * eps * ((sigma / cutoff_distance) ** 12 - (sigma / cutoff_distance) ** 6)
    return energy.sum() - energy_cutoff.sum()
