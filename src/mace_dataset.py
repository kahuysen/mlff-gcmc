"""Dataset voor de student: willekeurige CH4-posities in de lege IRMOF-1-cel met de MACE-interactie-energie.

Per punt: positie uniform in de cel, baseline-energie met energy_fast (united atom), all-atom CH4 met
willekeurige orientatie en het C-atoom exact op dat punt, dan
    E_int = E(rooster + CH4) - E(rooster) - E(CH4 los)
met een ASE-calculator (MACE op de HPC; voor een lokale test elke andere calculator).

Dit bestand heeft geen torch nodig: de calculator komt als argument binnen.
"""

import numpy as np
from ase import Atoms
from ase.build import molecule

EV_TO_K = 11604.518  # 1 eV / k_B in K
CH_BOND = 1.09  # A, C-H in ase.build.molecule("CH4")


def random_quaternion(rng):
    """Uniform verdeelde rotatie (Haar-maat) als eenheidsquaternion (w, x, y, z).

    Drie uniforme Euler-hoeken zijn NIET uniform over de rotaties (ophoping bij de polen).
    """
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def rotation_matrix(q):
    """Rotatiematrix van een eenheidsquaternion (w, x, y, z)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def place_ch4(position, rng):
    """All-atom CH4 (C + 4 H) met willekeurige orientatie en het C-atoom exact op position [A].

    Geeft (atoms, quaternion). Het C-atoom staat op het punt dat ook aan energy_fast wordt gegeven:
    in het united-atom model is de LJ-bol gecentreerd op de koolstof, en het massamiddelpunt van
    CH4 valt daarmee samen.
    """
    q = random_quaternion(rng)
    ch4 = molecule("CH4")  # C in de oorsprong, H op de tetraeder
    rel = ch4.get_positions() - ch4.get_positions()[0]
    rel = rel @ rotation_matrix(q).T
    atoms = Atoms(ch4.get_chemical_symbols(), positions=rel + np.asarray(position, dtype=float))
    assert atoms.get_chemical_symbols()[0] == "C"
    assert np.allclose(atoms.get_positions()[0], position)
    assert np.allclose(np.linalg.norm(atoms.get_positions()[1:] - atoms.get_positions()[0], axis=1), CH_BOND, atol=0.01)
    return atoms, q


def framework_plus_ch4(framework, ch4):
    """Rooster (met cel en pbc) plus de vijf CH4-atomen; cel en pbc komen van het rooster."""
    combined = framework + ch4
    combined.wrap()
    return combined


def reference_energies(calc, framework, ch4_box=20.0):
    """E(rooster) en E(CH4 los in een lege doos) in eV, een keer te berekenen."""
    fw = framework.copy()
    fw.calc = calc
    e_fw = fw.get_potential_energy()
    ch4 = molecule("CH4")
    ch4.set_cell([ch4_box] * 3)
    ch4.center()
    ch4.set_pbc(True)
    ch4.calc = calc
    e_ch4 = ch4.get_potential_energy()
    return e_fw, e_ch4


def interaction_energy(calc, framework, ch4, e_framework, e_ch4):
    """E_int [eV] = E(rooster + CH4) - E(rooster) - E(CH4 los)."""
    combined = framework_plus_ch4(framework, ch4)
    combined.calc = calc
    return combined.get_potential_energy() - e_framework - e_ch4


CSV_HEADER = "x,y,z,qw,qx,qy,qz,E_baseline_K,E_mace_K"
