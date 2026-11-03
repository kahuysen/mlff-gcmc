"""Dataset for the student: random CH4 positions in the empty IRMOF-1 cell with the MACE interaction energy.

Per point: position uniform in the cell, baseline energy with energy_fast (united atom), all-atom CH4 with
a random orientation and the C atom exactly on that point, then
    E_int = E(framework + CH4) - E(framework) - E(CH4 alone)
with an ASE calculator (MACE on the HPC; any other calculator for a local test).

This file does not need torch: the calculator comes in as an argument.
"""

import numpy as np
from ase import Atoms
from ase.build import molecule

EV_TO_K = 11604.518  # 1 eV / k_B in K
CH_BOND = 1.09  # A, C-H in ase.build.molecule("CH4")


def random_quaternion(rng):
    """Uniformly distributed rotation (Haar measure) as a unit quaternion (w, x, y, z).

    Three uniform Euler angles are NOT uniform over rotations (they pile up at the poles).
    """
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def rotation_matrix(q):
    """Rotation matrix of a unit quaternion (w, x, y, z)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def place_ch4(position, rng):
    """All-atom CH4 (C + 4 H) with a random orientation and the C atom exactly on position [A].

    Returns (atoms, quaternion). The C atom sits on the point that is also passed to energy_fast:
    in the united-atom model the LJ sphere is centred on the carbon, and the centre of mass of
    CH4 coincides with it.
    """
    q = random_quaternion(rng)
    ch4 = molecule("CH4")  # C at the origin, H on the tetrahedron
    rel = ch4.get_positions() - ch4.get_positions()[0]
    rel = rel @ rotation_matrix(q).T
    atoms = Atoms(ch4.get_chemical_symbols(), positions=rel + np.asarray(position, dtype=float))
    assert atoms.get_chemical_symbols()[0] == "C"
    assert np.allclose(atoms.get_positions()[0], position)
    assert np.allclose(np.linalg.norm(atoms.get_positions()[1:] - atoms.get_positions()[0], axis=1), CH_BOND, atol=0.01)
    return atoms, q


def framework_plus_ch4(framework, ch4):
    """Framework (with cell and pbc) plus the five CH4 atoms; cell and pbc come from the framework."""
    combined = framework + ch4
    combined.wrap()
    return combined


def reference_energies(calc, framework, ch4_box=20.0):
    """E(framework) and E(CH4 alone in an empty box) in eV, to be computed once."""
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
    """E_int [eV] = E(framework + CH4) - E(framework) - E(CH4 alone)."""
    combined = framework_plus_ch4(framework, ch4)
    combined.calc = calc
    return combined.get_potential_energy() - e_framework - e_ch4


CSV_HEADER = "x,y,z,qw,qx,qy,qz,E_baseline_K,E_mace_K"
