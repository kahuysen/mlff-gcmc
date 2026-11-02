"""Grand-canonical Monte Carlo van CH4 (united atom) in een rigide MOF met het baseline-krachtveld.

Frenkel & Smit 3e ed. par. 6.5 (algoritmen 17-19). mu, V en T liggen vast, N fluctueert;
de belading is <N>. Per stap een move met een random getal u: u < 1/3 verplaatsen,
1/3 <= u < 2/3 invoegen, anders verwijderen. Acceptatie met een tweede random getal.

Energieverschillen zijn lokaal: alleen de roosterterm van het bewogen/ingevoegde/verwijderde
molecuul plus zijn LJ-paren met de andere CH4's. Eenheden: energie in K, afstanden in A,
f in Pa, V in m^3.
"""

import numpy as np

from energy import energy_fast
from widom import AVOGADRO, U_TO_KG

K_B = 1.380649e-23  # J/K
R = 8.31446261815324  # J/(mol K)

CH4_EPS = 158.5  # K, Martin et al. 2001 (zie data/force_field/methane.json)
CH4_SIGMA = 3.72  # A
CUTOFF = 12.0  # A


def energy_of_molecule(ch4_position, others, positions, eps_pair, sigma_pair, L):
    """Energie [K] van een CH4 op ch4_position: rooster plus LJ met de CH4's in others.

    others bevat NIET het molecuul zelf (lege array (0, 3) als er geen andere zijn).
    """
    framework = energy_fast(ch4_position, positions, eps_pair, sigma_pair, L, cutoff_distance=CUTOFF)
    if len(others) == 0:
        return framework
    eps_ch4 = np.full(len(others), CH4_EPS)
    sigma_ch4 = np.full(len(others), CH4_SIGMA)
    guests = energy_fast(ch4_position, others, eps_ch4, sigma_ch4, L, cutoff_distance=CUTOFF)
    return framework + guests


def try_translate(index, ch4_positions, positions, eps_pair, sigma_pair, L, T, max_displacement, rng):
    """Verplaats molecuul index uniform in [-d, d]^3, wrap in de cel. Geeft (nieuwe positie, geaccepteerd, dU)."""
    old = ch4_positions[index]
    others = np.delete(ch4_positions, index, axis=0)
    new = (old + rng.uniform(-max_displacement, max_displacement, size=3)) % L
    dU = energy_of_molecule(new, others, positions, eps_pair, sigma_pair, L) - energy_of_molecule(
        old, others, positions, eps_pair, sigma_pair, L
    )
    accepted = rng.random() < min(1.0, np.exp(-dU / T))
    return new, accepted, dU


def try_insert(ch4_positions, positions, eps_pair, sigma_pair, L, T, f, rng):
    """Voeg een CH4 in op een uniforme positie. acc = min(1, fV/((N+1)kT) exp(-dU/T))."""
    new = rng.uniform(0.0, L, size=3)
    dU = energy_of_molecule(new, ch4_positions, positions, eps_pair, sigma_pair, L)
    N = len(ch4_positions)
    V = (L * 1e-10) ** 3
    acc = f * V / ((N + 1) * K_B * T) * np.exp(-dU / T)
    accepted = rng.random() < min(1.0, acc)
    return new, accepted, dU


def try_delete(ch4_positions, positions, eps_pair, sigma_pair, L, T, f, rng):
    """Verwijder een willekeurig CH4. acc = min(1, NkT/(fV) exp(-dU/T)) met dU = -U_i."""
    N = len(ch4_positions)
    if N == 0:
        return None, False, 0.0
    index = rng.integers(N)
    others = np.delete(ch4_positions, index, axis=0)
    dU = -energy_of_molecule(ch4_positions[index], others, positions, eps_pair, sigma_pair, L)
    V = (L * 1e-10) ** 3
    acc = N * K_B * T / (f * V) * np.exp(-dU / T)
    accepted = rng.random() < min(1.0, acc)
    return index, accepted, dU


def run_gcmc(
    positions, eps_pair, sigma_pair, L, T, f, n_steps, rng, max_displacement=1.5, n_equilibration=0, ch4_positions=None
):
    """GCMC vanaf een lege cel (of ch4_positions). Logt N na elke stap vanaf n_equilibration.

    Geeft een dict met n_history (productie), ch4_positions (eind), acceptatie per move
    (geaccepteerd / geprobeerd), en sum_dU: som van alle geaccepteerde dU [K], voor de
    boekhoudingscheck tegen total_energy.
    """
    if ch4_positions is None:
        ch4_positions = np.empty((0, 3))
    ch4_positions = np.array(ch4_positions, dtype=float)
    n_history = np.zeros(n_steps - n_equilibration, dtype=int)
    attempted = {"translate": 0, "insert": 0, "delete": 0}
    accepted_count = {"translate": 0, "insert": 0, "delete": 0}
    sum_dU = 0.0

    for step in range(n_steps):
        u = rng.random()
        if u < 1 / 3:
            if len(ch4_positions) > 0:
                attempted["translate"] += 1
                index = rng.integers(len(ch4_positions))
                new, accepted, dU = try_translate(
                    index, ch4_positions, positions, eps_pair, sigma_pair, L, T, max_displacement, rng
                )
                if accepted:
                    ch4_positions[index] = new
                    accepted_count["translate"] += 1
                    sum_dU += dU
        elif u < 2 / 3:
            attempted["insert"] += 1
            new, accepted, dU = try_insert(ch4_positions, positions, eps_pair, sigma_pair, L, T, f, rng)
            if accepted:
                ch4_positions = np.vstack([ch4_positions, new])
                accepted_count["insert"] += 1
                sum_dU += dU
        else:
            attempted["delete"] += 1
            index, accepted, dU = try_delete(ch4_positions, positions, eps_pair, sigma_pair, L, T, f, rng)
            if accepted:
                ch4_positions = np.delete(ch4_positions, index, axis=0)
                accepted_count["delete"] += 1
                sum_dU += dU
        if step >= n_equilibration:
            n_history[step - n_equilibration] = len(ch4_positions)

    acceptance = {k: accepted_count[k] / attempted[k] if attempted[k] else float("nan") for k in attempted}
    return {"n_history": n_history, "ch4_positions": ch4_positions, "acceptance": acceptance, "sum_dU": sum_dU}


def block_average(samples, n_blocks=10):
    """Gemiddelde en standaardfout via blokgemiddelden (zelfde recept als widom)."""
    samples = np.asarray(samples, dtype=float)
    n = len(samples) // n_blocks * n_blocks
    block_means = samples[:n].reshape(n_blocks, -1).mean(axis=1)
    return block_means.mean(), block_means.std(ddof=1) / np.sqrt(n_blocks)


def molecules_per_cell_to_mol_per_kg(n, masses):
    """n moleculen per cel -> mol/kg rooster. masses in u."""
    cell_mass_kg = np.sum(masses) * U_TO_KG
    return n / (AVOGADRO * cell_mass_kg)


# ---------------------------------------------------------------------------
# Trage referentie voor de boekhoudingscheck (bewust niet gevectoriseerd, los van energy_fast)
# ---------------------------------------------------------------------------


def _lj_shifted(r, eps, sigma, rc=CUTOFF):
    if r >= rc:
        return 0.0
    u = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
    u_rc = 4 * eps * ((sigma / rc) ** 12 - (sigma / rc) ** 6)
    return u - u_rc


def _min_image_distance(a, b, L):
    d = np.asarray(a) - np.asarray(b)
    d -= np.round(d / L) * L
    return float(np.sqrt(np.dot(d, d)))


def total_energy(ch4_positions, positions, eps_pair, sigma_pair, L):
    """Totale energie [K]: elk CH4 met elk roosteratoom, plus elk CH4-paar precies een keer."""
    U = 0.0
    for p in ch4_positions:
        for q, eps, sigma in zip(positions, eps_pair, sigma_pair):
            U += _lj_shifted(_min_image_distance(p, q, L), eps, sigma)
    n = len(ch4_positions)
    for i in range(n):
        for j in range(i + 1, n):
            U += _lj_shifted(_min_image_distance(ch4_positions[i], ch4_positions[j], L), CH4_EPS, CH4_SIGMA)
    return U


# ---------------------------------------------------------------------------
# Druk -> fugaciteit met Peng-Robinson
# ---------------------------------------------------------------------------

CH4_TC = 190.56  # K
CH4_PC = 45.99e5  # Pa
CH4_OMEGA = 0.011


def fugacity_coefficient(p, T, Tc=CH4_TC, pc=CH4_PC, omega=CH4_OMEGA):
    """phi = f/p voor een zuiver gas volgens Peng-Robinson (grootste reele wortel = gasfase)."""
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = (1 + kappa * (1 - np.sqrt(T / Tc))) ** 2
    a = 0.45724 * R**2 * Tc**2 / pc * alpha
    b = 0.07780 * R * Tc / pc
    A = a * p / (R * T) ** 2
    B = b * p / (R * T)
    roots = np.roots([1, -(1 - B), A - 3 * B**2 - 2 * B, -(A * B - B**2 - B**3)])
    Z = max(r.real for r in roots if abs(r.imag) < 1e-10)
    s2 = np.sqrt(2)
    ln_phi = Z - 1 - np.log(Z - B) - A / (2 * s2 * B) * np.log((Z + (1 + s2) * B) / (Z + (1 - s2) * B))
    return float(np.exp(ln_phi))


def fugacity(p, T):
    """Fugaciteit [Pa] van CH4 bij druk p [Pa] en temperatuur T [K]."""
    return p * fugacity_coefficient(p, T)
