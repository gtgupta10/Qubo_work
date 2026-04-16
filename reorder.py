
import os

N_PD = 32
N_ZN = 32
N_C  = 2
N_O  = 2

def parse_vasp(filepath):
    with open(filepath) as f:
        lines = f.readlines()

    scale   = float(lines[1])
    lattice = [list(map(float, lines[i].split())) for i in range(2, 5)]

    species_line = lines[5].split()
    counts_line  = list(map(int, lines[6].split()))

    atoms = []
    for sp, cnt in zip(species_line, counts_line):
        atoms.extend([sp] * cnt)

    idx = 7
    selective = lines[idx].strip().lower().startswith('s')
    if selective:
        idx += 1
    coord_type = lines[idx].strip()
    idx += 1

    coords = []
    for atom in atoms:
        parts = lines[idx].split()
        xyz   = list(map(float, parts[:3]))
        flags = parts[3:6] if selective and len(parts) >= 6 else ['F','F','F']
        coords.append((atom, xyz, flags))
        idx += 1

    return lattice, scale, coord_type, coords, selective


def write_vasp(filepath, lattice, scale, coord_type, ordered):
    with open(filepath, 'w') as f:
        f.write("Pd Zn C O\n")
        f.write(f"  {scale:.16f}\n")
        for row in lattice:
            f.write(f"    {row[0]:20.16f}  {row[1]:20.16f}  {row[2]:20.16f}\n")
        f.write("   Pd   Zn   C   O\n")
        f.write(f"   {N_PD}   {N_ZN}   {N_C}   {N_O}\n")
        f.write("Selective dynamics\n")
        f.write(f"{coord_type}\n")
        for atom, xyz, flags in ordered:
            fx, fy, fz = flags
            f.write(f"  {xyz[0]:20.16f}  {xyz[1]:20.16f}  {xyz[2]:20.16f}   {fx}   {fy}   {fz}\n")


def reorder(coords):
    pd_atoms = [(a, xyz, f) for a, xyz, f in coords if a == 'Pd']
    zn_atoms = [(a, xyz, f) for a, xyz, f in coords if a == 'Zn']
    c_atoms  = [(a, xyz, f) for a, xyz, f in coords if a == 'C']
    o_atoms  = [(a, xyz, f) for a, xyz, f in coords if a == 'O']

    assert len(pd_atoms) == N_PD, f"Expected {N_PD} Pd, got {len(pd_atoms)}"
    assert len(zn_atoms) == N_ZN, f"Expected {N_ZN} Zn, got {len(zn_atoms)}"
    assert len(c_atoms)  == N_C,  f"Expected {N_C} C,  got {len(c_atoms)}"
    assert len(o_atoms)  == N_O,  f"Expected {N_O} O,  got {len(o_atoms)}"

    def fix_slab(atom_list):
        # All slab atoms fully frozen regardless of original flags
        return [(a, xyz, ['F','F','F']) for a, xyz, _ in atom_list]

    def fix_co(atom_list):
        # CO: freeze XY, free Z only
        return [(a, xyz, ['F','F','T']) for a, xyz, _ in atom_list]

    return fix_slab(pd_atoms) + fix_slab(zn_atoms) + fix_co(c_atoms) + fix_co(o_atoms)


def process(folder):
    vasp_files = sorted(f for f in os.listdir(folder) if f.endswith('.vasp'))
    if not vasp_files:
        print("No .vasp files found in", folder)
        return

    out_folder = os.path.join(folder, "reordered")
    os.makedirs(out_folder, exist_ok=True)
    print(f"Processing {len(vasp_files)} files from {folder}/")

    ok, err = 0, 0
    for fname in vasp_files:
        fpath = os.path.join(folder, fname)
        try:
            lattice, scale, coord_type, coords, selective = parse_vasp(fpath)
            ordered = reorder(coords)
            write_vasp(os.path.join(out_folder, fname),
                       lattice, scale, coord_type, ordered)
            ok += 1
            if ok % 100 == 0:
                print(f"  ... {ok}/{len(vasp_files)}")
        except Exception as e:
            print(f"  ERROR {fname}: {e}")
            err += 1

    print(f"Done: {ok} written, {err} errors  →  {out_folder}/")


process("CO2_adsorbed")