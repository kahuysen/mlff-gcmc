"""Student: the same shifted LJ sum as energy.py, but with the eight CH4 pair parameters free,
fitted to MACE interaction energies (data/mace_dataset_irmof1_ch4.csv).

Parameter vector (flat, as scipy wants it):
    params = [eps_Zn, eps_O, eps_C, eps_H, sigma_Zn, sigma_O, sigma_C, sigma_H]   # pairs with CH4, K and A
"""

import numpy as np
from scipy.optimize import least_squares

from energy import energy_fast, lorentz_berthelot

ELEMENTS = ["Zn", "O", "C", "H"]
PARAM_NAMES = [f"eps_{e}" for e in ELEMENTS] + [f"sigma_{e}" for e in ELEMENTS]
LOWER = np.array([0.1] * 4 + [2.0] * 4)
UPPER = np.array([500.0] * 4 + [5.0] * 4)


def element_indices(symbols):
    """Index in ELEMENTS for every framework atom (Zn 0, O 1, C 2, H 3)."""
    return np.array([ELEMENTS.index(s) for s in symbols])


def baseline_params(lj_params, guest):
    """Start vector: Lorentz-Berthelot of each element with the guest, in the order of PARAM_NAMES."""
    pairs = [lorentz_berthelot(guest, lj_params[e]) for e in ELEMENTS]
    return np.array([p[0] for p in pairs] + [p[1] for p in pairs])


def params_to_pairs(params, element_index):
    """Eight pair parameters -> (eps_pair, sigma_pair), each of length len(element_index)."""
    params = np.asarray(params, dtype=float)
    return params[0:4][element_index], params[4:8][element_index]


def student_energies(params, dataset_positions, positions, element_index, L):
    """Student energy [K] at all dataset positions for one parameter vector."""
    eps_pair, sigma_pair = params_to_pairs(params, element_index)
    return np.array([energy_fast(pos, positions, eps_pair, sigma_pair, L) for pos in dataset_positions])


def residuals(params, dataset_positions, positions, element_index, L, E_target, weights=None):
    """sqrt(w) * (E_student - E_target); least_squares squares it, so the cost is sum w (dE)^2."""
    diff = student_energies(params, dataset_positions, positions, element_index, L) - E_target
    if weights is None:
        return diff
    return np.sqrt(weights) * diff


def boltzmann_weights(E_target, T_w):
    """Softened Boltzmann weights exp(-E/T_w). T_w > T makes the weighting less extreme."""
    return np.exp(-np.asarray(E_target) / T_w)


def fit_student(params_start, dataset_positions, positions, element_index, L, E_target, weights=None):
    """least_squares with bounds; returns the scipy result (x, cost, nfev, message)."""
    return least_squares(
        residuals,
        params_start,
        bounds=(LOWER, UPPER),
        args=(dataset_positions, positions, element_index, L, E_target, weights),
    )
