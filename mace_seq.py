#!/usr/bin/env python
"""
MACE sequential relaxation — bimolecular (2 CO per slab).
Reads reordered POSCARs from CO2_adsorbed/reordered/
E_ads = E_total - E_slab - 2×E_CO
Usage: python mace_seq_2CO.py
"""
import os, time, glob
from ase.io import read, write
from ase import Atoms
from mace.calculators import mace_mp
from ase.optimize import BFGSLineSearch

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_DIR   = "CO2_adsorbed"
OUTPUT_DIR  = "CO2_relaxed"
ENERGY_FILE = "energies_2CO.tsv"
SLAB_POSCAR = "clean_POSCAR"
FMAX        = 0.02    # eV/Å
MAX_STEPS   = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load calculator once ─────────────────────────────────────────────────────
print("Loading MACE ...")
t0   = time.time()
calc = mace_mp(model="medium-0b3", dispersion=False, default_dtype="float64")
print(f"  Loaded in {time.time()-t0:.1f}s")
print("="*70)

# ── Step 1: Relax bare slab ──────────────────────────────────────────────────
print("Step 1: Relaxing bare slab ...")
slab = read(SLAB_POSCAR)
slab.pbc = [True, True, False]
slab.calc = calc
opt = BFGSLineSearch(slab, logfile=os.path.join(OUTPUT_DIR, "slab_relax.log"))
opt.run(fmax=FMAX, steps=MAX_STEPS)
E_slab = slab.get_potential_energy()
write(os.path.join(OUTPUT_DIR, "slab_relaxed.vasp"), slab, format='vasp')
print(f"  E_slab = {E_slab:.6f} eV  |  steps = {opt.get_number_of_steps()}")
print("="*70)

# ── Step 2: Relax CO gas reference ───────────────────────────────────────────
print("Step 2: Relaxing CO gas molecule ...")
co_gas = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.15]],
               cell=[15, 15, 15], pbc=False)
co_gas.calc = calc
opt_co = BFGSLineSearch(co_gas, logfile=os.path.join(OUTPUT_DIR, "co_gas.log"))
opt_co.run(fmax=FMAX, steps=200)
E_CO = co_gas.get_potential_energy()
print(f"  E_CO   = {E_CO:.6f} eV  |  steps = {opt_co.get_number_of_steps()}")
print(f"  E_ads  = E_total - E_slab - 2×E_CO")
print("="*70)

# ── Step 3: Loop over all VASP files ─────────────────────────────────────────
vasp_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.vasp")))
n_files    = len(vasp_files)
print(f"Found {n_files} VASP files in {INPUT_DIR}/")
print(f"\n{'#':<4} {'File':<55} {'E_total':>14} {'E_ads':>10} {'Steps':>6} {'Time':>8} {'Status'}")
print("-"*110)

results = []
t_total = time.time()

for i, vasp_path in enumerate(vasp_files, 1):
    basename = os.path.basename(vasp_path)
    t0       = time.time()

    atoms = read(vasp_path)
    atoms.pbc = [True, True, False]
    atoms.calc = calc

    traj_path = os.path.join(OUTPUT_DIR, basename.replace(".vasp", ".traj"))
    log_path  = os.path.join(OUTPUT_DIR, basename.replace(".vasp", ".log"))

    opt = BFGSLineSearch(atoms, trajectory=traj_path, logfile=log_path)
    opt.run(fmax=FMAX, steps=MAX_STEPS)

    E_total   = atoms.get_potential_energy()
    nsteps    = opt.get_number_of_steps()
    elapsed   = time.time() - t0
    converged = nsteps < MAX_STEPS
    E_ads     = E_total - E_slab - 2 * E_CO      # 2 CO references
    status    = "OK" if converged else "!! NOT CONVERGED"

    write(os.path.join(OUTPUT_DIR, basename), atoms, format='vasp')
    results.append((basename, E_total, E_ads, nsteps, elapsed, converged))

    print(f"{i:<4} {basename:<55} {E_total:>14.6f} {E_ads:>10.4f} "
          f"{nsteps:>6} {elapsed:>7.1f}s {status}", flush=True)

# ── Step 4: Summary ──────────────────────────────────────────────────────────
print("\n" + "="*110)
print(f"{'Structure':<55} {'E_total (eV)':>14} {'E_ads (eV)':>11} {'Steps':>6} {'Time':>8}")
print("="*110)
best_fname = min(results, key=lambda x: x[2])[0]
for fname, e_total, e_ads, nsteps, elapsed, converged in results:
    marker = "  <- most stable"   if fname == best_fname else ""
    flag   = "  !! NOT CONVERGED" if not converged       else ""
    print(f"{fname:<55} {e_total:>14.6f} {e_ads:>11.4f} {nsteps:>6} {elapsed:>7.1f}s{marker}{flag}")

best = min(results, key=lambda x: x[2])
print("="*110)
print(f"\n*  Strongest adsorption : {best[0]}")
print(f"   E_ads = {best[2]:.4f} eV")
print(f"   Total wall time : {time.time()-t_total:.1f}s")
print(f"   Relaxed VASP    -> ./{OUTPUT_DIR}/")
print(f"   Energy table    -> ./{ENERGY_FILE}")

# ── Step 5: Write energy table ───────────────────────────────────────────────
results.sort(key=lambda x: x[0])
with open(ENERGY_FILE, "w") as f:
    f.write("# filename\tE_total_eV\tE_ads_eV\tsteps\ttime_s\tconverged\n")
    f.write(f"# E_slab  = {E_slab:.6f} eV\n")
    f.write(f"# E_CO    = {E_CO:.6f} eV\n")
    f.write(f"# E_ads   = E_total - E_slab - 2*E_CO\n")
    for fname, e_total, e_ads, nsteps, elapsed, converged in results:
        f.write(f"{fname}\t{e_total:.6f}\t{e_ads:.6f}\t{nsteps}\t{elapsed:.1f}\t{converged}\n")