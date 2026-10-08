# O on Pt(111): QUBO matrices for QpiAI-Opt

Three versions of the same 96-variable QUBO for oxygen adsorption on a Pt(111) surface, with exact reference answers to check a solver against.

## What the problem is

- **96 binary variables**, one per adsorption site in a 4×4 surface cell. `x_i = 1` means an O atom sits on site `i`.
- **Objective:** minimize `E(x) = xᵀ W x`, in eV.
- **Site types by index** (full list with coordinates in `sites.csv`):

  | Indices | Site type | Count |
  |---|---|---|
  | 0–15 | ontop | 16 |
  | 16–63 | bridge | 48 |
  | 64–79 | fcc hollow | 16 |
  | 80–95 | hcp hollow | 16 |

- **Coverage** = (number of O) / 16, in monolayers (ML).

## The files

| File | Contents | Use it for |
|---|---|---|
| `QUBO_mace-omat-0_dftcorr.npy` | MACE-OMAT pair interactions, diagonal corrected to DFT | **Main matrix: physically validated** |
| `QUBO_mace-omat-0.npy` | MACE-OMAT, no correction | Comparison |
| `QUBO_mace-mp-0b3.npy` | Older MACE model | Comparison only: it misses the O–O repulsion, so its physics is wrong |
| `sites.csv` | Site index, type, x/y position (Å) | Same ordering for all three matrices |
| `expected_results.json` | All reference answers below, plus optimal site lists | Machine-readable checks |

All three are 96 × 96, float64, in eV.

## How the matrix is stored: read this before loading it

`W` is **upper-triangular**. The lower triangle is all zeros.

| Entry | Meaning |
|---|---|
| `W[i,i]` | energy cost of one O on site `i` |
| `W[i,j]`, `i < j` | the **full** interaction between O atoms on sites `i` and `j`, counted once |
| `W[i,j] = 10.0` | **excluded pair**: the two sites are under 2 Å apart, so they can never both be occupied. Keep these entries. |

Both of these load it correctly:

- **Dictionary form:** `{(i,i): W[i,i]}` plus `{(i,j): W[i,j]}` for every `i < j`, each pair entered once.
- **Symmetric-matrix form:** `Q = (W + W.T) / 2`. The diagonal is unchanged and each pair is split as half into `(i,j)` and half into `(j,i)`, so `xᵀQx = xᵀWx`.

Do **not** halve `W[i,j]` and then also enter it only once; that counts every interaction at half strength. An earlier project had exactly this bug.

Quick self-check (should print `4.921  -1.079`):

```python
import numpy as np
W = np.load("QUBO_mace-omat-0_dftcorr.npy")
x = np.zeros(96); x[[64, 66, 72, 74]] = 1          # 4 O atoms in the p(2x2) pattern
U = 0.75
print(round(x @ W @ x, 3), round(x @ (W - 2 * U * np.eye(96)) @ x, 3))
```

## Which constraint to use

**Key point:** at U = 0 every diagonal entry is positive (+1.16 to +2.97 eV). So with no constraint and no potential, the best answer is always the **empty surface** (`x = 0`, `E = 0`). The same goes for an "at most M atoms" constraint (the slack-variable form used in the CO/PdZn work): it also returns the empty surface here. Use one of the two options below.

### Option A, recommended: electrode potential U, no constraint

Pick a potential `U` (in V vs the reversible hydrogen electrode) and minimize

```
E(x) = xᵀ (W − 2U·I) x
```

That's just `W` with `2U` subtracted from every diagonal entry. **No constraint, slack variables or penalty are needed:** the potential decides how many O atoms adsorb, and the 10 eV entries already enforce the exclusions.

### Option B: a fixed number of O atoms, k

Minimize `xᵀWx` subject to `Σ x_i = k`. As a penalty this is

```
E(x) = xᵀ W x + P (Σ x_i − k)²
```

which expands into the QUBO as:
- diagonal: `+ P(1 − 2k)`
- every pair `i < j`: `+ 2P`
- a constant `+ P k²`, which doesn't affect the optimum but does shift the reported energy

**P = 5 eV is enough for all three matrices.** It's checked against the reference answers: adding or removing one O costs at most about 2.2 eV, so violating the constraint is never worth it. Always confirm `Σx = k` in the returned solution.

## Expected results

All values were checked two ways: exact enumeration, and simulated annealing over all 96 sites. They agree to the meV.

**Optima are not unique.** The surface is symmetric, so each optimum has many equivalent copies with identical energy. **Compare energies, not site indices.**

### Option A: the optimum at a given potential U

| Matrix | U (V) | Number of O | Optimal E (eV) | Arrangement |
|---|---|---|---|---|
| omat-0_dftcorr | 0.50 | 0 | 0.000 | empty surface |
| omat-0_dftcorr | **0.75** | **4** | **−1.079** | **p(2×2)** |
| omat-0_dftcorr | 0.92 | 6 | −2.607 | denser fcc |
| omat-0 | 0.50 | 0 | 0.000 | empty surface |
| omat-0 | 0.85 | 4 | −0.937 | p(2×2) |
| omat-0 | 1.05 | 6 | −2.754 | denser fcc |
| mp-0b3 | 0.50 | 0 | 0.000 | empty surface |
| mp-0b3 | 0.93 | 4 | −0.575 | compact cluster (wrong physics) |
| mp-0b3 | 1.00 | 6 | −1.224 | compact cluster |

### Option B: the optimum for k O atoms, `min xᵀWx` with `Σx = k`

These are the energies of `xᵀWx`, **without** the penalty's constant `P k²`:

| k | Coverage (ML) | omat-0_dftcorr | omat-0 | mp-0b3 |
|---|---|---|---|---|
| 1 | 0.0625 | 1.1637 | 1.3991 | 1.6387 |
| 2 | 0.125 | 2.3718 | 2.8427 | 3.3280 |
| 3 | 0.1875 | 3.6242 | 4.3306 | 5.0678 |
| **4** | **0.25** | **4.9210** | **5.8628** | **6.8653** |
| 5 | 0.3125 | 6.7091 | 7.8863 | 8.7998 |
| 6 | 0.375 | 8.4329 | 9.8455 | 10.7756 |
| 7 | 0.4375 | 10.4183 | 12.0665 | 12.8090 |
| 8 | 0.5 | 12.3395 | 14.2231 | 14.8930 |

### The physics check: 4 O atoms

For both OMAT matrices, the optimum at k = 4 (or at U = 0.75 V for the corrected matrix) must be the known **p(2×2)** pattern: **4 O atoms, all on fcc sites (indices 64–79), each pair exactly 5.61 Å apart**. One example is sites `[64, 66, 72, 74]`.

The old `mp-0b3` matrix instead gives a compact cluster of fcc O about 2.81 Å apart. That's why it's labelled wrong physics.

To check distances yourself: the cell vectors are a₁ = (11.2289, 0) Å and a₂ = (5.6144, 9.7245) Å. The cell is periodic, so use the shortest periodic image (it's a 60° cell, so check the neighbouring images, not just rounded coordinates).

### Coverage vs potential (main matrix, for reference)

| U (V vs RHE) | Number of O | Coverage |
|---|---|---|
| 0 – 0.58 | 0 | clean surface |
| 0.59 – 0.64 | 1 – 3 | isolated O |
| **0.65 – 0.87** | **4** | **0.25 ML, p(2×2) plateau** |
| 0.88 – 0.97 | 6 | 0.375 ML |
| ≥ 0.98 | ≥ 8 | ≥ 0.5 ML; beyond 8 O was not enumerated, so there's no reference value |

A solver that reproduces this staircase (scan U and record how many O come back) is getting the physics right.

## Where these numbers come from

- **Single-O energies (diagonal):** per site, referenced to water: H₂O(g) → O* + H₂(g). At potential U each O gains −2U eV, from H₂O → O* + 2(H⁺ + e⁻).
- **Pair energies (off-diagonal):** O–O interactions from MACE relaxations of every symmetry-distinct pair of sites.
- **DFT correction (dftcorr):** the diagonal is shifted per site type to Quantum ESPRESSO (PBE) values. Bridge sites are not corrected.
- **Validation:** MACE-OMAT reproduces the DFT nearest-neighbour O–O repulsion; the older mp-0b3 model does not.
