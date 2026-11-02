"""Taak 5: energie van de lege IRMOF-1-cel met MACE-MP-0 medium + D3 op de GPU, en de tijd per evaluatie.

Draaien op de HPC:  module swap cluster/joltik && cd $VSC_DATA/cusp-mace && sbatch run_gpu_joltik.sh test_mace.py
(of accelgor met run_gpu.sh; joltik had op 2 nov 3 min wachttijd, accelgor 3,5 uur)
"""
import os
import time

import ase.io
import torch
from mace.calculators import mace_mp


def reset_all(calc):
    """ASE cachet resultaten per calculator. Met dispersion=True is calc een SumCalculator
    (MACE + D3) zonder eigen reset, dus reset de onderliggende calculators."""
    if hasattr(calc, "reset"):
        calc.reset()
    else:  # BaseCalculator (bv. SumCalculator): zelfde effect als Calculator.reset()
        calc.atoms = None
        calc.results = {}
    for sub in getattr(getattr(calc, "mixer", None), "calcs", []):
        reset_all(sub)


cif = os.path.join(os.environ["VSC_DATA"], "cusp-mace", "data", "IRMOF-1.cif")
atoms = ase.io.read(cif, format="cif")
print(f"{len(atoms)} atomen, cel {atoms.cell.lengths().round(3)} A, pbc {atoms.pbc}")

t0 = time.time()
calc = mace_mp(model="medium", dispersion=True, device="cuda", default_dtype="float64")
print(f"calculator geladen in {time.time() - t0:.1f} s")
print("calculator:", type(calc).__name__, "->", [type(c).__name__ for c in getattr(getattr(calc, "mixer", None), "calcs", [])])
atoms.calc = calc

# Opwarmer: de eerste aanroep compileert kernels en is niet representatief.
t0 = time.time()
E0 = atoms.get_potential_energy()
torch.cuda.synchronize()
print(f"E(lege cel) = {E0:.4f} eV = {E0 / len(atoms):.4f} eV/atoom   (eerste aanroep {time.time() - t0:.2f} s)")

# Drie getimede evaluaties. reset_all wist de cache, anders geeft ASE het oude getal terug in 0 s.
times = []
for i in range(3):
    reset_all(calc)
    t0 = time.time()
    E = atoms.get_potential_energy()
    torch.cuda.synchronize()
    times.append(time.time() - t0)
    print(f"evaluatie {i + 1}: E = {E:.4f} eV, {times[-1]:.3f} s")
print(f"tijd per evaluatie (gemiddelde van 3): {sum(times) / 3:.3f} s")
assert min(times) > 0.01, "evaluatie kwam uit de cache, reset_all werkt niet"

# Kleine sanity check: een CH4 all-atom losse molecule, zelfde calculator.
from ase.build import molecule
ch4 = molecule("CH4")
ch4.set_cell([20.0, 20.0, 20.0])
ch4.set_pbc(True)
ch4.calc = calc
reset_all(calc)
t0 = time.time()
E_ch4 = ch4.get_potential_energy()
torch.cuda.synchronize()
print(f"E(CH4 los, 20 A doos) = {E_ch4:.4f} eV in {time.time() - t0:.3f} s")
