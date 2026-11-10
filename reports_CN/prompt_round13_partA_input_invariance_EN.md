# Round 13 — Part A: is the designed trigger's information independent of the input shape?

Run only after every gate of Part 0 has passed. Same rules as Part 0 (§0 there).

## 1. The hypothesis being tested

With constant-threshold SoD the number of transmissions is tied to how far the output travels, so the input cannot be chosen freely: paper 1 gained only 0.70 nat by designing the input. With a designed window table the facing edge follows the predicted output and the number of transmissions is set by the price, not by the output's motion. Hypothesis ("separation"): **once the table is designed for the input, the information of the record is the classical full-data information of that input times an input-independent factor**,

```
I_table(input) ≈ I_full(input) + c,       c the same for every input,
```
where `I = log det J`, `J_full = σ⁻² Σ_k h_k h_kᵀ` (every sample transmitted) and `I_table` is the information of the designed table at the fixed budget `B = 316.8921168493721`. If it holds, input design under a designed trigger reduces to classical input design. Under constant SoD (threshold re-tuned to the same budget) no such relation is expected.

This part only tests the hypothesis on six inputs; it proves nothing.

## 2. Fixed quantities

System, frequencies `dr.w`, `a0 = 1`, noise `s`, `K = 5000`, total power `Σ a_m² = 1`, budget `B` (the same number for every input), action set `eval13.edge_set()` (10 candidate edge offsets, minimum distance 2σ, 100 actions), criterion: the 9×9 `log det J` (`p_dim = 9`), tangent `gamma_fn = Design11.gamma_from_J`.

## 3. Inputs (from `inputs13.json`)

| name | what it is | why it is in the test |
|---|---|---|
| `U` | uniform amplitudes, zero phases | baseline (Round 11 reproduces exactly) |
| `P5514` | paper 1's best budget-feasible input under constant SoD: amplitudes within 1 % of uniform, optimised phases | under constant SoD it is +0.70 nat over `U`; same `I_full` as `U` |
| `SCH` | uniform amplitudes, Schroeder phases | phase-only change; same `I_full` as `U` |
| `LO` | power (0.40, 0.30, 0.20, 0.10) on (ω₁…ω₄), zero phases | larger output swing (1347 σ against 1144 σ) |
| `HI` | power (0.10, 0.20, 0.30, 0.40), zero phases | smaller output swing (869 σ); same `I_full` as `LO` |
| `N` | `config.json` input `N`: paper 1's best input without budget (424 transmissions at Δ*) | skewed amplitudes and optimised phases |

`LO` and `HI` have the same power multiset, hence (up to cross terms) the same `I_full`, but very different output motion: the pair is the sharpest test.

## 4. What to compute for each input (one process per input, `R13_INPUT=<name>`)

1. `I_full` (`common13.full_information`).
2. Constant SoD at Δ* (`des.sod_reference()`): `E[N]`, `S`, `I` — shows how far this input is from the budget with the nominal threshold.
3. Matched constant SoD (`common13.matched_sod(B)`): `M*`, residual `E[N] − B`, `S`, `I_SoD` on σ/24 and σ/120.
4. Designed table (`common13.design(edge_set(), Design11.gamma_from_J, logdet9, 9, 'edge')`): scans `j0`, `j1`, best table `T*`, its `I`, `E[N]`, `S` on σ/24 and σ/120, certificate `UB'` and `gap`, shape statistics.
5. Derived, with the σ/120 values:
   - `c_tab = I_table − I_full`,  `c_sod = I_SoD − I_full`;
   - `gain = I_table − I_SoD` (table against matched SoD for the same input);
   - information per transmission `(S − 1)/(E[N] − 1)` for the table and for matched SoD.

`U` is not special-cased: it runs through the same code, and its numbers must equal Round 11's (gate G0 already checks this; re-assert `I_table(U)` on σ/24 = 207.35521730257756).

## 5. Cross-input summary (`run13.py partA_summary`, after all six are done)

One table, one row per input: `I_full`, constant SoD at Δ* (`E[N]`, `I`), matched SoD (`M*`, `E[N]`, `I`, `c_sod`), table (`E[N]`, `S`, `I`, `c_tab`, certificate gap), `gain`.

Then:
- `range_c_tab = max − min` of `c_tab` over the six inputs; the same for `c_sod`.
- `S` ratio: `max_i |S_table(i)/S_table(U) − 1|`.
- Rank agreement: order the inputs by `I_full` and by `I_table`, treating values within 0.1 nat as tied; report both orders and whether they agree. The same for `I_SoD`.
- Pairwise checks: `I_table(SCH) − I_table(U)`, `I_table(P5514) − I_table(U)`, `c_tab(LO) − c_tab(HI)`; and the same three differences for matched SoD.

One figure (`fig_partA.png`, one column, font ≥ 8 pt): x = `I_full − I_full(U)`, y = `I − I(U)`; two series, designed table (filled markers) and matched SoD (open markers), one marker per input, labelled; the line y = x dotted. Caption states what to look at: points of the table on the line y = x mean separation.

## 6. Verdict (fixed before running)

- **H1** `max_i |S_table(i)/S_table(U) − 1| <= 0.03`.
- **H2** `range_c_tab <= 0.5` nat.
- **H3** `|I_table(SCH) − I_table(U)| <= 0.2` and `|I_table(P5514) − I_table(U)| <= 0.2` and `|c_tab(LO) − c_tab(HI)| <= 0.5` nat.
- Resolution condition: every reported table has certificate `gap <= 0.5` nat (otherwise differences of that size cannot be told from design noise; `common13.design` already adds a third scan when needed). If after the third scan some gap is still > 0.5, say so and mark the verdict "unresolved for input …".

Verdict: **separation supported** if H1, H2, H3 all hold; **not supported** if `range_c_tab > 1.5` nat; otherwise **partial** (report which of H1–H3 failed and by how much). `c_sod` and the matched-SoD differences are reported as contrast, not as gates.

Do not write causes. "Supported" means the six inputs behaved as the hypothesis predicts, nothing more.

## 7. Predictions (write to `round13/predictions_partA.md` and commit before any Part A computation)

```
- H1 holds: S_table within ±3 % of S_table(U) for all six inputs.
- H2 holds: range of c_tab <= 0.5 nat.  Point predictions on sigma/24, from c(U) = 207.355 − 219.009 = −11.65:
  I_table ≈ 207.35 (P5514, SCH), 206.38 (LO, HI), 206.76 (N), each ±0.3.
- H3 holds.
- Constant SoD is not separable: range of c_sod >= 0.7 nat, and |c_sod(LO) − c_sod(HI)| >= 0.5 nat.
- Matched SoD: I_SoD(P5514) − I_SoD(U) ∈ [0.5, 0.9] nat (paper 1's +0.70 at nearly the same budget).
- Certificate gap <= 0.3 nat for every input after two scans.
- gain = I_table − I_SoD ∈ [9, 11.5] nat for every input.
Scoring: a line is a hit iff every stated number falls in its range.
```

## 8. Checks

1. Every forward: mass conservation (1e-12 on σ/24, 1e-10 on σ/120), non-negativity, zero mass at the grid boundary.
2. Every scan: monotonicity check of `Scan11` passes; forward score of each final table against its fixed-policy backward value ≤ 1e-12 relative (`eval13.backward`).
3. Every reported table: `E[N]_{120} <= B`; `I(T) <= UB'`.
4. Matched SoD accepted (`|E[N]_{120} − B| <= 1`) for every input; if not, stop for that input and report.
5. `U` reproduces Round 11 (see §4).
6. For `LO` and `HI`: `|I_full(LO) − I_full(HI)| <= 0.01` nat (they should be equal up to cross terms; if not, the construction of the inputs is wrong).

## 9. Outputs

- `round13/results/partA_<name>.json` per input; `round13/results/partA_summary.json`; packed best tables `table_A_<name>.npz`; `fig_partA.png`.
- `reports_CN/report_round13_partA_CN.md` (+ EN copy): (1) one-sentence verdict; (2) the cross-input table; (3) H1–H3 with numbers; (4) the constant-SoD contrast; (5) predictions hit/miss; (6) checks; (7) figure with one descriptive paragraph; (8) DECISION list; (9) what was not done.
- Runtime guide: per input about 15 min matched SoD, 30 min for two scans (each DP ≈ 45–60 s, each σ/24 forward ≈ 20–25 s, more for the larger grids of `LO`, `N`), a few minutes σ/120; six inputs, four processes in parallel.
