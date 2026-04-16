"""
By
Geet Gupta 

site_find_2CO.py  (v7 — all 4560 pairs, no distance filter)
─────────────────────────────────────────────────────────────────────────────
Places TWO CO molecules on every combination of 2 distinct adsorption sites
from the 96-site set.  96C2 = 4560 POSCARs total.  No distance filtering,
no label-based deduplication.

Site labeling uses the v5 image-expansion approach (3×3 tiling) so
bridge/hollow sites near cell edges always find their true neighbours.
─────────────────────────────────────────────────────────────────────────────
"""

from ase.io import read
from ase import Atoms
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.surface import Slab
import numpy as np
from collections import Counter
import itertools
import csv
import os

# ═══════════════════════════════════════════════════════════════════════════
#  USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════
C_O_BOND   = 1.15  # Å — C≡O bond length
C_HEIGHT   = 1.8   # Å — C atom height above topmost surface atom
O_HEIGHT   = C_HEIGHT + C_O_BOND
OUTPUT_DIR = "CO2_adsorbed"
# ═══════════════════════════════════════════════════════════════════════════

EXPECTED_SINGLE = {
    "ontop_Pd"        :  8,
    "ontop_Zn"        :  8,
    "bridge_Pd-Pd"    :  8,
    "bridge_Pd-Zn"    : 32,
    "bridge_Zn-Zn"    :  8,
    "hollow_Pd-Pd-Zn" : 16,
    "hollow_Pd-Zn-Zn" : 16,
}

# ─── 1. Load slab ────────────────────────────────────────────────────────────

slab_ase     = read('clean_POSCAR')
slab_ase.pbc = [True, True, False]

adaptor    = AseAtomsAdaptor()
pmg_struct = adaptor.get_structure(slab_ase)

pmg_slab = Slab(
    lattice            = pmg_struct.lattice,
    species            = pmg_struct.species,
    coords             = pmg_struct.frac_coords,
    miller_index       = (1, 0, 0),
    oriented_unit_cell = pmg_struct,
    shift              = 0,
    scale_factor       = np.eye(3, dtype=int),
    reorient_lattice   = True,
)

finder    = AdsorbateSiteFinder(pmg_slab)
all_sites = finder.find_adsorption_sites(
    distance         = 2.0,
    symm_reduce      = 0,
    near_reduce      = 0.01,
    no_obtuse_hollow = False,
)

surf_sites     = finder.surface_sites
surf_pos_prim  = np.array([s.coords for s in surf_sites])
surf_spec_prim = np.array([str(s.specie) for s in surf_sites])

# ─── 2. Expand to 3×3 periodic images ────────────────────────────────────────

a_vec = slab_ase.cell[0]
b_vec = slab_ase.cell[1]

img_pos, img_spec = [], []
for na in (-1, 0, 1):
    for nb in (-1, 0, 1):
        shift = na * a_vec + nb * b_vec
        img_pos.append(surf_pos_prim + shift)
        img_spec.append(surf_spec_prim)

surf_pos_ext  = np.vstack(img_pos)
surf_spec_ext = np.concatenate(img_spec)

# ─── 3. Neighbour search helpers ─────────────────────────────────────────────

def dist_xy_ext(site_xy):
    return np.linalg.norm(surf_pos_ext[:, :2] - site_xy, axis=1)

def get_nearest_n_ext(site_coords, n):
    dxy = dist_xy_ext(site_coords[:2])
    idx = np.argsort(dxy)[:n]
    return [(dxy[i], surf_spec_ext[i]) for i in idx]

def get_bridge_pair_ext(site_coords):
    site_xy  = site_coords[:2]
    dxy      = dist_xy_ext(site_xy)
    d_min    = dxy.min()
    cand_idx = np.where(dxy <= d_min * 1.5)[0]
    if len(cand_idx) < 2:
        cand_idx = np.argsort(dxy)[:6]

    best_d_mid, best_mm = np.inf, np.inf
    best_pair = tuple(cand_idx[:2])

    for i in range(len(cand_idx)):
        for j in range(i + 1, len(cand_idx)):
            ii, jj = cand_idx[i], cand_idx[j]
            mid    = 0.5 * (surf_pos_ext[ii, :2] + surf_pos_ext[jj, :2])
            d_mid  = np.linalg.norm(mid - site_xy)
            mm     = max(dxy[ii], dxy[jj])
            if (d_mid < best_d_mid - 1e-6 or
                    (abs(d_mid - best_d_mid) < 1e-6 and mm < best_mm)):
                best_d_mid, best_mm = d_mid, mm
                best_pair = (ii, jj)

    ii, jj = best_pair
    return [(dxy[ii], surf_spec_ext[ii]), (dxy[jj], surf_spec_ext[jj])]

# ─── 4. Labeling ─────────────────────────────────────────────────────────────

def label_site(site_type, site_coords):
    if site_type == "ontop":
        neighbors = get_nearest_n_ext(site_coords, 1)
    elif site_type == "bridge":
        neighbors = get_bridge_pair_ext(site_coords)
    else:
        neighbors = get_nearest_n_ext(site_coords, 3)
    species_sorted = sorted(sp for _, sp in neighbors)
    parts          = "-".join(species_sorted)
    return f"{site_type}_{parts}", neighbors

# ─── 5. Collect & label all 96 sites ─────────────────────────────────────────

labeled_sites = []
for site_type in ["ontop", "bridge", "hollow"]:
    for coords in all_sites.get(site_type, []):
        label, neighbors = label_site(site_type, np.array(coords))
        labeled_sites.append({
            "idx":       len(labeled_sites) + 1,
            "type":      site_type,
            "label":     label,
            "coords":    np.array(coords),
            "neighbors": neighbors,
        })

# ─── 6. Validate single-site counts ──────────────────────────────────────────

summary   = Counter(s["label"] for s in labeled_sites)
total_s   = sum(summary.values())
all_match = True

print("\n" + "="*70)
print(f"  SINGLE-SITE VALIDATION  ({total_s} sites found)")
print("="*70)
print(f"  {'Label':<30} {'Found':>6}  {'Expected':>8}  {'Match?':>10}")
print("  " + "-"*56)
for label in sorted(set(list(summary.keys()) + list(EXPECTED_SINGLE.keys()))):
    found    = summary.get(label, 0)
    expected = EXPECTED_SINGLE.get(label, "—")
    if isinstance(expected, int):
        ok    = found == expected
        match = "✓" if ok else "✗ ← MISMATCH"
        if not ok: all_match = False
    else:
        match = "(unexpected)"; all_match = False
    print(f"  {label:<30} {found:>6}  {str(expected):>8}  {match:>10}")
print("  " + "-"*56)
print(f"  {'TOTAL':<30} {total_s:>6}  {96:>8}  {'✓' if total_s==96 else '✗'}")
status = "✓  All single-site counts match." if (all_match and total_s==96) \
         else "✗  Mismatch — pair labels may be wrong."
print(f"  {status}\n")

# ─── 7. PBC-aware inter-site distance ────────────────────────────────────────

cell_2d = slab_ase.cell[:2, :2]
inv_2d  = np.linalg.inv(cell_2d)

def pbc_dist(c1, c2):
    d    = np.array(c1[:2]) - np.array(c2[:2])
    frac = d @ inv_2d.T
    frac -= np.round(frac)
    return float(np.linalg.norm(frac @ cell_2d))

# ─── 8. Build ALL 4560 pairs ─────────────────────────────────────────────────

n            = len(labeled_sites)
total_combos = n * (n - 1) // 2   # 4560
print(f"Building all {total_combos} pairs (96C2, no distance filter) …")

all_pairs = []
for i, j in itertools.combinations(range(n), 2):
    s1, s2 = labeled_sites[i], labeled_sites[j]
    dist   = pbc_dist(s1["coords"], s2["coords"])
    all_pairs.append({
        "s1": s1, "s2": s2,
        "dist": dist,
        "label_key": tuple(sorted([s1["label"], s2["label"]])),
    })

all_pairs.sort(key=lambda p: (p["s1"]["label"], p["s2"]["label"], p["dist"]))
for k, p in enumerate(all_pairs, 1):
    p["pair_idx"] = k

assert len(all_pairs) == total_combos, f"Expected {total_combos}, got {len(all_pairs)}"
print(f"  → {len(all_pairs)} pairs built  ✓")

# ─── 9. Print pair-type summary ──────────────────────────────────────────────

pair_type_summary = Counter(p["label_key"] for p in all_pairs)

print(f"\n{'='*78}")
print(f"  PAIR-TYPE SUMMARY  ({len(pair_type_summary)} distinct label combinations, "
      f"{len(all_pairs)} total pairs)")
print("="*78)
print(f"  {'Site-1 label':<30}  {'Site-2 label':<30}  {'Count':>6}  {'%':>6}")
print("  " + "-"*76)
for (l1, l2), cnt in sorted(pair_type_summary.items()):
    pct = 100 * cnt / len(all_pairs)
    print(f"  {l1:<30}  {l2:<32}  {cnt:>6}  {pct:>5.1f}%")
print("  " + "-"*76)
print(f"  {'TOTAL':<64}  {len(all_pairs):>6}  100.0%")

# Distance statistics
dists = np.array([p["dist"] for p in all_pairs])
print(f"\n  Inter-site distance stats across all {len(all_pairs)} pairs:")
print(f"    min = {dists.min():.3f} Å   max = {dists.max():.3f} Å   "
      f"mean = {dists.mean():.3f} Å   median = {np.median(dists):.3f} Å")

# ─── 10. Ask user ────────────────────────────────────────────────────────────

print()
answer = input(
    f"Proceed? Write {len(all_pairs)} POSCAR files + manifest CSV "
    f"to ./{OUTPUT_DIR}/  [yes/no]: "
).strip().lower()
if answer not in ("yes", "y"):
    print("Aborted.")
    exit()

# ─── 11. Write POSCARs + CSV manifest ────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

surf_z_ase = slab_ase.get_positions()[:, 2].max()
z_C        = surf_z_ase + C_HEIGHT
z_O        = surf_z_ase + O_HEIGHT

print(f"\nSurface z (ASE)  = {surf_z_ase:.3f} Å")
print(f"C above surface  = {C_HEIGHT:.2f} Å   =>  z_C = {z_C:.3f} Å")
print(f"O above surface  = {O_HEIGHT:.2f} Å   =>  z_O = {z_O:.3f} Å")
print(f"Output folder    : ./{OUTPUT_DIR}/\n")

def make_co(x, y):
    return Atoms("CO", positions=[[x, y, z_C], [x, y, z_O]])

csv_path = os.path.join(OUTPUT_DIR, "pairs_manifest.csv")
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "pair_idx",
        "site1_idx", "site1_label", "site1_x", "site1_y",
        "site2_idx", "site2_label", "site2_x", "site2_y",
        "dist_ang", "filename",
    ])

    for p in all_pairs:
        slab_copy = slab_ase.copy()
        for s in (p["s1"], p["s2"]):
            slab_copy += make_co(s["coords"][0], s["coords"][1])

        l1    = p["s1"]["label"].replace(" ", "_")
        l2    = p["s2"]["label"].replace(" ", "_")
        fname = f"pair{p['pair_idx']:04d}_{l1}__{l2}_d{p['dist']:.1f}A.vasp"
        write(os.path.join(OUTPUT_DIR, fname), slab_copy, format="vasp")

        writer.writerow([
            p["pair_idx"],
            p["s1"]["idx"], p["s1"]["label"],
            f"{p['s1']['coords'][0]:.4f}", f"{p['s1']['coords'][1]:.4f}",
            p["s2"]["idx"], p["s2"]["label"],
            f"{p['s2']['coords'][0]:.4f}", f"{p['s2']['coords'][1]:.4f}",
            f"{p['dist']:.4f}", fname,
        ])

        if p["pair_idx"] % 500 == 0 or p["pair_idx"] == len(all_pairs):
            print(f"  ... written {p['pair_idx']:>4}/{len(all_pairs)}")

print(f"\nDone!")
print(f"  {len(all_pairs)} POSCAR files  →  ./{OUTPUT_DIR}/")
print(f"  Manifest CSV             →  {csv_path}")
