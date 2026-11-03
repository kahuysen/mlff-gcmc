"""Student: dezelfde geshifte LJ-som als energy.py, maar met de acht CH4-paarparameters vrij,
gefit op MACE-interactie-energieen (data/mace_dataset_irmof1_ch4.csv).

Parametervector (plat, zo wil scipy het):
    params = [eps_Zn, eps_O, eps_C, eps_H, sigma_Zn, sigma_O, sigma_C, sigma_H]   # paren met CH4, K en A
"""

import numpy as np
from scipy.optimize import least_squares

from energy import energy_fast, lorentz_berthelot

ELEMENTS = ["Zn", "O", "C", "H"]
PARAM_NAMES = [f"eps_{e}" for e in ELEMENTS] + [f"sigma_{e}" for e in ELEMENTS]
LOWER = np.array([0.1] * 4 + [2.0] * 4)
UPPER = np.array([500.0] * 4 + [5.0] * 4)


def element_indices(symbols):
    """Per roosteratoom de index in ELEMENTS (Zn 0, O 1, C 2, H 3)."""
    return np.array([ELEMENTS.index(s) for s in symbols])


def baseline_params(lj_params, guest):
    """Startvector: Lorentz-Berthelot van elk element met de gast, in de volgorde van PARAM_NAMES."""
    pairs = [lorentz_berthelot(guest, lj_params[e]) for e in ELEMENTS]
    return np.array([p[0] for p in pairs] + [p[1] for p in pairs])


def params_to_pairs(params, element_index):
    """Acht paarparameters -> (eps_pair, sigma_pair), elk van lengte len(element_index)."""
    params = np.asarray(params, dtype=float)
    return params[0:4][element_index], params[4:8][element_index]


def student_energies(params, dataset_positions, positions, element_index, L):
    """Studentenergie [K] op alle dataset-posities voor een parametervector."""
    eps_pair, sigma_pair = params_to_pairs(params, element_index)
    return np.array([energy_fast(pos, positions, eps_pair, sigma_pair, L) for pos in dataset_positions])


def residuals(params, dataset_positions, positions, element_index, L, E_target, weights=None):
    """sqrt(w) * (E_student - E_target); least_squares kwadrateert zelf, dus de kost is sum w (dE)^2."""
    diff = student_energies(params, dataset_positions, positions, element_index, L) - E_target
    if weights is None:
        return diff
    return np.sqrt(weights) * diff


def boltzmann_weights(E_target, T_w):
    """Afgevlakte Boltzmann-gewichten exp(-E/T_w). T_w > T maakt de weging minder extreem."""
    return np.exp(-np.asarray(E_target) / T_w)


def fit_student(params_start, dataset_positions, positions, element_index, L, E_target, weights=None):
    """least_squares met grenzen; geeft het scipy-resultaat (x, cost, nfev, message)."""
    return least_squares(
        residuals,
        params_start,
        bounds=(LOWER, UPPER),
        args=(dataset_positions, positions, element_index, L, E_target, weights),
    )
