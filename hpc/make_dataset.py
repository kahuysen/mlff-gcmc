"""Taak 6: MACE-dataset van CH4-interactie-energieen in de lege IRMOF-1-cel.

HPC:   module swap cluster/joltik && cd $VSC_DATA/cusp-mace && \
       sbatch --time=02:00:00 run_gpu_joltik.sh make_dataset.py --n 3000 --seed 6 --out mace_dataset.csv
Lokaal (pijplijntest zonder torch):  python hpc/make_dataset.py --calc lj --n 20 --out /tmp/test.csv

Schrijft de CSV elke --flush punten opnieuw weg, zodat een afgebroken job niets kwijt is.
Punten met baseline-energie boven --max-baseline (K) worden overgeslagen (CH4 op een roosteratoom)
en alleen geteld.
"""

import argparse
import os
import sys
import time

import ase.io
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.path.join(HERE, "..", "src"), os.path.join(HERE, "src")):
    if os.path.isdir(cand):
        sys.path.insert(0, cand)
        break

from energy import energy_fast, load_force_field, load_framework, pairwise_lj_params  # noqa: E402
from gcmc import CH4_EPS, CH4_SIGMA  # noqa: E402
from mace_dataset import CSV_HEADER, EV_TO_K, interaction_energy, place_ch4, reference_energies  # noqa: E402


def find(rel):
    for base in (os.path.join(HERE, ".."), HERE):
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(rel)


def make_calculator(name):
    if name == "mace":
        from mace.calculators import mace_mp

        return mace_mp(model="medium", dispersion=True, device="cuda", default_dtype="float64")
    if name == "lj":  # alleen om de pijplijn te testen, fysisch zinloos
        from ase.calculators.lj import LennardJones

        return LennardJones(sigma=3.0, epsilon=0.01, rc=6.0)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="aantal MACE-punten (overgeslagen overlaps niet meegeteld)")
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--out", default="mace_dataset.csv")
    ap.add_argument("--calc", default="mace", choices=["mace", "lj"])
    ap.add_argument("--max-baseline", type=float, default=10_000.0, help="K; hoger = overlap, overslaan")
    ap.add_argument("--flush", type=int, default=50)
    args = ap.parse_args()

    cif = find("data/IRMOF-1.cif")
    positions, symbols, L, masses = load_framework(cif)
    lj_params = load_force_field(find("data/force_field/generic_dreiding_uff_mixing_rules.def"))
    eps_pair, sigma_pair = pairwise_lj_params(symbols, lj_params, guest=(CH4_EPS, CH4_SIGMA))
    framework = ase.io.read(cif, format="cif")
    print(f"{len(framework)} roosteratomen, L = {L:.3f} A, calculator = {args.calc}", flush=True)

    t0 = time.time()
    calc = make_calculator(args.calc)
    e_fw, e_ch4 = reference_energies(calc, framework)
    print(f"E(rooster) = {e_fw:.4f} eV, E(CH4 los) = {e_ch4:.4f} eV, klaar in {time.time() - t0:.1f} s", flush=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    n_skipped = 0
    t0 = time.time()
    while len(rows) < args.n:
        pos = rng.random(3) * L
        e_base = energy_fast(pos, positions, eps_pair, sigma_pair, L)
        if e_base > args.max_baseline:
            n_skipped += 1
            continue
        ch4, q = place_ch4(pos, rng)
        e_int = interaction_energy(calc, framework, ch4, e_fw, e_ch4) * EV_TO_K
        rows.append([*pos, *q, e_base, e_int])
        if len(rows) % args.flush == 0 or len(rows) == args.n:
            np.savetxt(args.out, np.array(rows), delimiter=",", header=CSV_HEADER, comments="", fmt="%.6f")
            dt = time.time() - t0
            print(
                f"{len(rows):5d}/{args.n} punten, {n_skipped} overgeslagen, {dt / len(rows):.3f} s/punt, "
                f"laatste: E_base = {e_base:9.1f} K, E_mace = {e_int:9.1f} K",
                flush=True,
            )
    print(f"klaar: {len(rows)} punten in {time.time() - t0:.0f} s, {n_skipped} overlaps overgeslagen -> {args.out}")


if __name__ == "__main__":
    main()
