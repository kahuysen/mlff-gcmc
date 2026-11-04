"""Task 5: energy of the empty IRMOF-1 cell with MACE-MP-0 medium + D3 on the GPU, and the time per evaluation.

Run on the HPC:  module swap cluster/joltik && cd $VSC_DATA/mlff-gcmc && sbatch run_gpu_joltik.sh test_mace.py
(or accelgor with run_gpu.sh; on 2 Nov joltik had a 3 min queue, accelgor 3.5 h)
"""
import os
import time

import ase.io
import torch
from mace.calculators import mace_mp


def reset_all(calc):
    """ASE caches results per calculator. With dispersion=True, calc is a SumCalculator
    (MACE + D3) without its own reset, so reset the underlying calculators."""
    if hasattr(calc, "reset"):
        calc.reset()
    else:  # BaseCalculator (e.g. SumCalculator): same effect as Calculator.reset()
        calc.atoms = None
        calc.results = {}
    for sub in getattr(getattr(calc, "mixer", None), "calcs", []):
        reset_all(sub)


cif = os.path.join(os.environ["VSC_DATA"], "mlff-gcmc", "data", "IRMOF-1.cif")
atoms = ase.io.read(cif, format="cif")
print(f"{len(atoms)} atoms, cell {atoms.cell.lengths().round(3)} A, pbc {atoms.pbc}")

t0 = time.time()
calc = mace_mp(model="medium", dispersion=True, device="cuda", default_dtype="float64")
print(f"calculator loaded in {time.time() - t0:.1f} s")
print("calculator:", type(calc).__name__, "->", [type(c).__name__ for c in getattr(getattr(calc, "mixer", None), "calcs", [])])
atoms.calc = calc

# Warm-up: the first call compiles kernels and is not representative.
t0 = time.time()
E0 = atoms.get_potential_energy()
torch.cuda.synchronize()
print(f"E(empty cell) = {E0:.4f} eV = {E0 / len(atoms):.4f} eV/atom   (first call {time.time() - t0:.2f} s)")

# Three timed evaluations. reset_all clears the cache, otherwise ASE returns the old number in 0 s.
times = []
for i in range(3):
    reset_all(calc)
    t0 = time.time()
    E = atoms.get_potential_energy()
    torch.cuda.synchronize()
    times.append(time.time() - t0)
    print(f"evaluation {i + 1}: E = {E:.4f} eV, {times[-1]:.3f} s")
print(f"time per evaluation (mean of 3): {sum(times) / 3:.3f} s")
assert min(times) > 0.01, "evaluation came from the cache, reset_all does not work"

# Small sanity check: an isolated all-atom CH4 molecule, same calculator.
from ase.build import molecule
ch4 = molecule("CH4")
ch4.set_cell([20.0, 20.0, 20.0])
ch4.set_pbc(True)
ch4.calc = calc
reset_all(calc)
t0 = time.time()
E_ch4 = ch4.get_potential_energy()
torch.cuda.synchronize()
print(f"E(CH4 alone, 20 A box) = {E_ch4:.4f} eV in {time.time() - t0:.3f} s")
