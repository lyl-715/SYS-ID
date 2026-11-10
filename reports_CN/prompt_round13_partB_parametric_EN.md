# Round 13 — Part B: a criterion for which the classical optimal input is not the uniform one

Run after Part A is complete (it reuses Part A's stored `J` matrices and its infrastructure). Same rules as Part 0.

## 1. Why this part exists

In Part A the criterion is the 9×9 `log det J` of the frequency-response values under a total-power constraint; for that criterion the classical optimal input is (up to cross terms) the uniform multisine that is already in use, so Part A can show separation but cannot show a gain from designing the input. Here the criterion is the information about the **physical parameters** of the transfer function, for which the classical optimal input concentrates the power. The questions:

1. Does separation also hold for this criterion?
2. Is "classical optimal input + table designed for it" the best of the tested combinations, and how much does it gain over "uniform input + table" and over constant SoD?

## 2. Parametric criterion

Model `G(s; b, τ) = b/(1 + τ s)²`, true value `ϑ* = (b, τ) = (2, 1)` (this is `dr.G`). The regression parameter is `θ(ϑ) = [G(0), Re G(jω₁), Im G(jω₁), …, Im G(jω₄)]` (order of `dr.regressors`). Jacobian `D = ∂θ/∂ϑ ∈ ℝ^{9×2}` at `ϑ*`:

- row 0: `[1, 0]`  (`G(0) = b`);
- for each frequency `ω_m`: `g_b = 1/(1 + jω_m τ)²`, `g_τ = −2 b jω_m/(1 + jω_m τ)³`; rows `[Re g_b, Re g_τ]` and `[Im g_b, Im g_τ]`.

Since `x_k = h_kᵀ θ(ϑ)`, the Fisher information about `ϑ` of any record with FRF information `J` is `M = Dᵀ J D` (2×2). Criterion: `I^par(J) = log det(Dᵀ J D)`.

Implementation in `round13/par13.py`:
- `D` as above. **Check B1**: compare with central differences of `θ(ϑ)` (step 1e-6): max abs error ≤ 1e-9; and `θ(ϑ*)` equals `[dr.G(0), Re dr.Gj, Im dr.Gj]` exactly. (Independent computation before this prompt: error 2.9e-11; `|g_τ|` = 0.2049, 0.0443, 0.0275, 0.0262 at the four frequencies — the time constant is seen almost only through ω₁.)
- `crit_par(J) = slogdet(D.T @ J @ D)[1]`.
- `gamma_par(J_ref)`: `M_ref = D.T @ J_ref @ D`; `d = HK @ D` (K×2); `gamma_k = d_k M_ref⁻¹ d_kᵀ / s²` (solve with Cholesky of `M_ref`); return `(gamma, mean(gamma[1:]), gamma[0])`. This is the tangent of `log det(Dᵀ J D)` at `J_ref`: `log det M ≤ log det M_ref + tr(M_ref⁻¹ M) − 2` and `tr(M_ref⁻¹ Dᵀ J D) = Σ_k w_k gamma_k`. **Check B2**: for the reference record's weights `W` (from its forward), `gamma @ W = 2` to 1e-9 relative.
- The design is `common13.design(edge_set(), gamma_par, crit_par, 2, 'par')`: `crit_par` decides the better of `T0`, `T1`; certificate `UB = crit_par(J(T0)) − 2 + min_λ dual`, and the `B + 0.25` variant; third scan if the gap exceeds 0.5 nat. `Scan11` itself is unchanged: it still logs the 9×9 `logdet` and `dmD`, which are ignored here; feasibility and the choice of the final table of a scan (largest `F` among feasible) work on the parametric `gamma`.

## 3. Inputs

`U`, `LO`, `HI` from Part A, plus `PAR`, the classical optimal input for the parametric criterion:

```
PAR = argmax over p on the simplex {Σ p_m = 1, p_m ≥ 0.01} of  log det( Dᵀ J_full(p) D ),   zero phases,
```
with `J_full(p) = diag(c) Q diag(c)`, `Q = H₁ᵀ H₁ / s²` for unit amplitudes and zero phases (`H₁ = dr.regressors(ones(4), zeros(4), dr.t)`), `c = (1, √p₁, √p₁, √p₂, √p₂, …)`. Evaluate on the exhaustive grid of multiples of 0.01 (156 849 points) and take the maximiser. **Check B3**: the maximiser is `p = (0.97, 0.01, 0.01, 0.01)` with value 49.2509 against 48.0196 for the uniform input (computed independently; agreement to 1e-4). Also report the unconstrained maximiser on the grid including zeros, for information only.

The lower bound `p_m ≥ 0.01` is fixed (DECISION 13-B1): it keeps the 9×9 matrices positive definite (the scan and the SoD reference take their log-determinant) and keeps all four frequencies in the model. Register `PAR` in `inputs13.json` (`p = [0.97, 0.01, 0.01, 0.01]`, zero phases) and run the Part-0 sanity line for it (`Σ a²`, output range, largest per-step motion ≤ 4.6 σ, `G`, `I_full`); for identification: range 1743.2 σ, largest step 4.10 σ, `G` = 49518, 9×9 `I_full` = 202.4069.

For reference, the classical values `I^par_full = log det(Dᵀ J_full D)` computed independently: 48.0196 (`U`), 48.4359 (`LO`), 47.2803 (`HI`), 49.2509 (`PAR`); **check B4**: agreement to 1e-4.

## 4. What to compute

For each of `U`, `LO`, `HI`, `PAR` (one process per input):

1. `I^par_full`.
2. Matched constant SoD: for `U`, `LO`, `HI` reuse the `J` of Part A's matched SoD (σ/120) → `I^par_SoD = crit_par(J)`; for `PAR` run `common13.matched_sod(B)` first. `PAR`'s output swings 1743 σ (largest per-step motion 4.10 σ, `G` = 49518), so its matched threshold is large: a quick simulation before this prompt gave 315 transmissions at `M = 850` (35.4 σ), so expect `M*` near 845–850; `ConstRows` handles `M > 959` by clipping if it is ever needed (Part 0 §2.4).
3. Table designed for the parametric criterion: `T*_par`, `I^par_table` on σ/24 and σ/120, `E[N]_{120} ≤ B`, `S`, certificate, shape statistics.
4. Cross-evaluation: Part A's best table for the same input (designed for the 9×9 criterion; not available for `PAR`) on the parametric criterion: `I^par(T*_FRF) = crit_par(J(T*_FRF))` from the stored σ/120 `J`.

Summary (`run13.py partB_summary`):

- `c^par_tab = I^par_table − I^par_full` and `c^par_sod = I^par_SoD − I^par_full` per input; their ranges over the four inputs.
- Best input for the designed table: `argmax I^par_table`.
- Gains, all on σ/120:
  - `G1 = I^par_table(PAR) − I^par_table(U)` — what the input adds once the trigger is designed; classical prediction `I^par_full(PAR) − I^par_full(U) = 1.231`.
  - `G2 = I^par_table(U) − I^par_SoD(U)` — what the trigger adds at the uniform input.
  - `G4 = I^par_SoD(PAR) − I^par_SoD(U)` — what the input adds under constant SoD at the same budget.
  - `G3 = I^par_table(PAR) − I^par_SoD(U)` — joint design against the baseline; and `G3 − (G2 + G4)`.
  - `G5 = I^par_table(U) − I^par(T*_FRF(U))` — what designing the table for the right criterion adds.
- One figure (`fig_partB.png`): same layout as Part A's (x = `I^par_full − I^par_full(U)`, y = `I^par − I^par(U)`, table and matched SoD, four inputs, line y = x).

## 5. Verdict (fixed before running)

- **H4** range of `c^par_tab` over the four inputs ≤ 0.4 nat.
- **H5** `PAR` is the best input for the designed table: `I^par_table(PAR) ≥ max over the other three − 0.1`.
- Resolution condition: certificate gap ≤ 0.5 nat for every reported table.

"Separation supported for the parametric criterion" if H4 and H5 hold; "not supported" if the range in H4 exceeds 1.2 nat or `PAR` is worse than another input by more than 0.3 nat; otherwise "partial". `G1`–`G5` are reported, not judged.

## 6. Predictions (write to `round13/predictions_partB.md` and commit before any Part B computation)

```
- B3: PAR = (0.97, 0.01, 0.01, 0.01).
- H4 holds (range of c^par_tab <= 0.4 nat); H5 holds.
- G1 ∈ [0.9, 1.6] nat (classical prediction 1.23).
- G2 ∈ [2.0, 4.5] nat.
- G5 ∈ [0.2, 2.0] nat.
- G4: |G4| <= 1.0 nat and G3 − (G2 + G4) >= −0.3 nat (no stronger claim).
- Matched SoD for PAR: M* ∈ [830, 870].
- Certificate gap <= 0.3 nat for every input.
Scoring: a line is a hit iff every stated number falls in its range.
```

## 7. Checks

B1–B4 above, plus the forward/backward, conservation, budget and certificate checks of Part A §8 (items 1–4) applied to every Part B table, with the certificate inequality `crit_par(J(T)) ≤ UB'`.

## 8. Outputs

- `round13/results/partB_<name>.json`, `partB_summary.json`, packed tables `table_B_<name>.npz`, `fig_partB.png`, `par13.py`.
- `reports_CN/report_round13_partB_CN.md` (+ EN copy): (1) one-sentence verdict; (2) the classical optimum and how it was found; (3) cross-input table (`I^par_full`, matched SoD, table, `c` values, certificate gaps); (4) H4, H5; (5) `G1`–`G5` with one sentence each saying what the number compares; (6) predictions hit/miss; (7) checks; (8) figure and one descriptive paragraph; (9) DECISION list; (10) not done.
- Runtime guide: four inputs × two scans (≈ 30 min each input), one matched-SoD search for `PAR` (≈ 15 min), σ/120 re-evaluations; four processes in parallel.

## 9. Not in this round

Proving separation; other systems or noise levels; the 884-action set; model mismatch; changing the frequencies; optimising phases for the parametric criterion; joint optimisation of input and table (only the tested combinations are compared).
