# Review: "A quantum eigensolver framework for assessing solvent effects of an SN2 reaction"

Reviewer notes on the **main manuscript only** (SI excluded, per request). Page numbers refer to the PDF as provided.

---

## 1. Spelling / grammar / wording highlights

These are line-level edits — treat as suggested track-changes.

| Page | Current text | Issue | Suggested fix |
|---|---|---|---|
| 2 | "conventional exchange correlation functionals" | Missing hyphen | "exchange-correlation functionals" |
| 3 | "gas-phase embedding treatment alone is insufficient for describing the majority of chemically relevant processes, which occur not in isolation but in the presence of a solvent environment that fundamentally shapes reactivity..." | One 70+ word run-on sentence | Split into two sentences for readability |
| 3 | "Kaliakin et al. have demonstrated..." / "Castaldo et al. have performed..." | Present-perfect tense is a bit informal/inconsistent for citing prior work | Consider simple past: "demonstrated," "performed" (ACS style generally prefers past tense for cited results) |
| 3 | "This work demonstrates the molecular simulations with SMD implicit solvation." | Awkward article usage | "This work demonstrates molecular simulations with SMD implicit solvation." |
| 17 | "The resource needs alignment with fault-tolerant estimates reported in prior research, and the quantum error-correction layer... is the primary expense..." | Subject–verb mismatch: "resource needs" (plural noun phrase) + "alignment" reads as if "needs" is a verb, creating ambiguity | Reword: "These resource requirements align with fault-tolerant estimates reported in prior research..." |
| 19 | "Even for the smallest active space (4e,4o); they already require ∼10⁶ s" | Semicolon used where a comma is grammatically correct | "...(4e,4o), they already require..." |
| 14 | "This shows that the embedding scheme is not an ad hoc starting point for a reaction-coordinate study, but a consistent description of the full profile." | "ad hoc starting point" is an odd descriptor for a "scheme" | Reword: "...is not merely valid at an isolated point along the reaction coordinate, but provides a consistent description of the full profile." |
| 15–16 | "8.36 kcal mol⁻¹at low ε to 22.72 kcal mol⁻¹in water" | Missing spaces before "at"/"in" (likely a LaTeX/PDF export artifact, but check the source .tex) | Insert space after unit superscript throughout — this pattern recurs several times (pp. 15, 16, 19) |
| 18 | "the low-error "Majorana model" considerably lowers..." | Inconsistent quoting style — "Majorana model" is in quotes here but not elsewhere (e.g., p. 19 "a problem that takes minutes on a 'Majorana' device") | Standardize: either always quote the informal shorthand or never |
| throughout | "SN2" rendered inconsistently as "S$_N$2" vs "SN2" in running text | Cosmetic/typesetting | Confirm the subscript renders correctly in the final typeset PDF (currently reads "SN2" without subscript in body text extraction) |
| 21 | "the steady parameterized research and development of better algorithms" | "parameterized" is an odd modifier for "research" — likely meant "continued/incremental" | Reword: "the steady, incremental research and development of better algorithms" |

**General**: several sentences exceed 50–60 words (e.g., p.3 intro paragraph, p.17 resource-estimate paragraph). Consider breaking these up for readability — not a correctness issue, but will help reviewers/readers parse the argument.

---

## 2. Technical gaps / clarity issues in Computational Methods

These are substantive points a referee is likely to raise.

### 2.1 Geometry/solvation workflow consistency
> 📍 **Page 11–12** — "Computational methods" section (IRC/TS optimization + "the quantum embedding calculations were carried out in SMD implicit solvation...")

- The IRC and transition-state search are performed **in vacuum** (B3LYP/6-31+G*), and solvent effects are then added only as **single-point SMD corrections on the gas-phase geometries**. This is a common approximation, but it is never explicitly flagged as one. A referee will ask: how much would the barrier shift if the TS/IRC were re-optimized *in* each solvent, rather than solvent-corrected post hoc? Recommend adding a sentence acknowledging this as an approximation (or a justification for why gas-phase geometries are expected to be adequate here).

### 2.2 Missing thermochemical corrections
> 📍 **Pages 13–17, Figures 3 and 4** — "Results and Discussion" (all ΔE / ΔE‡ values, e.g., "8.56 kcal mol⁻¹", "13.79 kcal mol⁻¹" on p. 14)

- All reported quantities are **purely electronic** energies (ΔE, ΔE‡). There is no mention of zero-point energy, thermal, or entropic corrections. Since SN2 barriers are conventionally discussed as ΔG‡ or ΔH‡ in the experimental literature (including ref. 38, which is cited for exactly this reason), the absence of any thermochemical correction — or at least an explicit statement that only electronic energies are being compared — is a gap worth flagging.

### 2.3 No direct experimental/literature benchmark
> 📍 **Page 14** ("DFT places the transition state at only 8.56 kcal mol⁻¹, whereas all of the correlated treatments raise it to 13.79 kcal mol⁻¹...") and **Figure 3**

- The gas-phase CCSD-in-DFT/VQE-in-DFT barrier (13.79 kcal/mol) is presented without comparison to literature high-level values for Cl⁻ + CH₃Cl (well-studied benchmark system; various CCSD(T)/CBS and G2+ values exist in the literature, typically ~13–14 kcal/mol electronic barrier). Adding this comparison would substantially strengthen the credibility of the embedding+VQE approach.

### 2.4 Basis set adequacy for an anionic system
> 📍 **Page 11** ("DFT methods with the B3LYP functional and the Pople-style 6-31+G* basis set family were employed...") and **Page 12** ("The environment subsystem was treated at the RKS/B3LYP/6-31+G* level of theory throughout.")

- 6-31+G* provides diffuse functions only on non-hydrogen atoms and only a single set of diffuse s/p functions. For an anion-molecule complex (Cl⁻···CH₃Cl), this is a fairly modest basis by current standards — no discussion of basis-set sensitivity, BSSE (counterpoise correction), or a comparison against a larger basis (e.g., aug-cc-pVDZ/TZ) is given. Worth at least a remark on expected basis-set error, since this could be comparable in magnitude to the correlation-recovery effects being highlighted (~5 kcal/mol DFT vs. correlated gap).

### 2.5 Active-space consistency along the IRC
> 📍 **Page 5, Eq. 5** (SPADE gap criterion) and **Page 12** ("The chlorine and the carbon atoms directly involved in bond breaking and bond formation were selected as the active subsystem... Orbital partitioning... was accomplished using the SPADE approach")

- The SPADE partitioning criterion (largest singular-value gap, Eq. 5) is applied per-geometry. The manuscript states this is "parameter-free," which is true in the sense that no numeric cutoff is chosen by hand — but it does not confirm that the **same number of orbitals** (i.e., same mA, consistent with the nominal (4e,4o)/(6e,6o)/(8e,8o) labels) is selected automatically and consistently at *every* point along the IRC and in every solvent. If mA were to change between two adjacent IRC points, this would introduce a discontinuity in the energy profile. A brief statement confirming this was checked (or how disagreements, if any, were handled) would close this gap.

### 2.6 No quantitative VQE-vs-CCSD comparison table
> 📍 **Pages 14–15, Figure 3** ("the methodology performs equally well whether the active subsystem is treated with the gold-standard CCSD or with VQE...", "enlarging the active space... changes the barrier by less than 0.05 kcal mol⁻¹")

- The claim that VQE "performs equally well" as CCSD and that active-space enlargement changes the barrier "by less than 0.05 kcal/mol" is stated narratively but not backed by a table of raw numbers (e.g., VQE−CCSD energy differences at each IRC point/active space/solvent). Given this is one of the two central claims of the paper, a small results table (or SI table referenced from the main text) reporting mean absolute deviation between VQE and CCSD would make this quantitative rather than qualitative.

### 2.7 "Intermediate values" claim in Fig. 4 discussion
> 📍 **Page 16, Figure 4** ("the VQE-in-DFT curves cluster at intermediate values, reaching 21.90 kcal mol⁻¹")

- The text states VQE-in-DFT barriers "cluster at intermediate values" between DFT and CCSD-in-DFT. Using the numbers given (DFT 14.18, VQE 21.90, CCSD 22.72 kcal/mol in water), VQE is only 0.82 kcal/mol below CCSD but 7.72 kcal/mol above DFT — i.e., VQE sits much closer to CCSD than to DFT, not literally "intermediate" (midpoint-like). Recommend rewording to something like "VQE-in-DFT barriers track closely with CCSD-in-DFT, lying just below it," which is a more accurate characterization and actually a *stronger* result for the paper's argument (VQE ≈ CCSD).

### 2.8 VQE optimizer/statistics not fully specified
> 📍 **Pages 12–13** ("Parameter optimization was carried out using the Adam optimizer with a step size of α = 0.008 and a maximum of 400 iterations. Convergence was declared when the absolute change in energy... fell below 10⁻⁵ Eh.")

- Adam optimizer settings are given, but there's no mention of:
  - Whether multiple random seeds/initializations were tried, and whether results were stable across them.
  - Whether any runs failed to converge within 400 iterations along the IRC (this can happen near the TS where the surface is flatter/harder).
  - Circuit-level statistics (gate count, circuit depth, number of variational parameters) for the actual noiseless-simulator VQE runs — this is distinct from the FTQC resource-estimate section on pp. 17–20 (which addresses hypothetical future hardware, not the simulations actually performed).

### 2.9 Frozen-core / basis details for the correlated step not stated
> 📍 **Page 12** ("The active subsystem was treated at two levels of theory: CCSD... and VQE... both implemented within PySCF and PennyLane, respectively.")

- It isn't explicit whether a frozen-core approximation was used for Cl core electrons in the CCSD/VQE active-space treatment, nor whether the same AO basis is used for the active subsystem as for the environment (RKS/B3LYP/6-31+G*). This is likely "yes" by convention, but should be stated explicitly rather than implied.

### 2.10 Environment level of theory and self-interaction error
> 📍 **Page 8–9, Eq. 11 and Eq. 15** (embedding potential / solvated KS Hamiltonian) cross-referenced with **Page 14** ("This gap reflects the well-known tendency of DFT to underestimate SN2 barriers, which has been attributed to the delocalization (self-interaction) error...")

- The environment subsystem is treated with B3LYP throughout, including for the global SMD/RKS step that generates the embedding potential (Eq. 11) and reaction field (Eq. 15). Since the paper's own argument (p. 14) is that B3LYP underestimates the SN2 barrier due to delocalization/self-interaction error, it's worth a sentence addressing whether/how this same error could leak into the embedding potential felt by the active subsystem (via Vemb, which depends on DB from the same B3LYP density) — i.e., is the correction achieved by the embedded WF treatment complete, or only partial, given that the environment density itself is not error-free?

### 2.11 Figure 5 axis units are easy to misread
> 📍 **Page 17, Figure 5 and its caption** ("Resource estimates were obtained as a function of active space dimension, ranging from 8 to 24 spin-orbitals" vs. Fig. 5 x-axis "4–12")

- Fig. 5's x-axis is "Active Space Dimension" running from 4–12, while the main text (p. 17) says resource estimates span "8 to 24 spin-orbitals." The caption clarifies "4 and 6 represent (4e,4o) and (6e,6o)," but doesn't extend this mapping to 8, 10, 12. Recommend either relabeling the axis directly in terms of spin-orbitals (8–24) to match the running text, or explicitly stating in the caption that the active-space dimension axis is defined as (Ne,No) with N = 4...12, so spin-orbitals = 2N.

---

## 3. Overall assessment

**Strengths**
- Clear, well-motivated combination of PBE embedding + SMD implicit solvation + VQE — a genuinely useful extension of prior gas-phase VQE-in-DFT and solvent-only (PCM) VQE work (refs. 34–36).
- The gas-phase vs. solvent-phase comparison (Figs. 3–4) is a clean, chemically intuitive result (DFT underestimates the barrier; correlated/embedded methods and VQE agree closely with CCSD).
- The FTQC resource-estimate section (Figs. 5–6) is a nice forward-looking addition connecting near-term demonstration to hardware roadmap discussion.
- Theory section (PBE derivation, Eqs. 1–16) is rigorous and self-contained.

**Main weaknesses to address before submission**
1. The paper's two central quantitative claims (VQE ≈ CCSD; consistent solvent trend across methods) are currently supported only narratively — add a numeric comparison table.
2. No thermochemical corrections and no comparison to experimental/high-level literature barriers — this will likely be requested by a referee for a benchmark system as well-studied as Cl⁻ + CH₃Cl.
3. Approximations in the workflow (gas-phase geometries + post-hoc solvation; basis set choice for an anionic system) should be explicitly acknowledged as limitations, even briefly, rather than left implicit.
4. A few wording/characterization issues (esp. §2.7 "intermediate values") slightly undersell what is actually a strong result (VQE tracking CCSD closely).

None of these are fatal — they're the kind of gaps a referee will flag in first review, so addressing them proactively should smooth the review process. Recommend closing items 2.6, 2.7, and the thermochemistry point (2.2) first, as those are the lowest-effort, highest-impact fixes.
