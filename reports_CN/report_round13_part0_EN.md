---
title: "Round 13, Part 0 report: infrastructure for other inputs, and regression gates"
date: "2026-11-10"
---

English copy of `report_round13_part0_CN.md`. Where the two differ, the Chinese version is the report of record.

**Prompt.** `reports_CN/prompt_round13_part0_infrastructure_EN.md` (with `reports_CN/README_round13.md`). Branch `round13-input-trigger`, created from `round11-edge-actions` at `1b88f01`. `claude/optimistic-newton-xv5jvl` (`3314e7d`), `round10-upper-bound` (`2b1af1b`) and `round11-edge-actions` (`1b88f01`) were not touched.
- `c65c7cb`: the four `.md` files of the package go into `reports_CN/` (the first commit of the round).
- `ff3debf`: code, gate results and logs (`round13/`).
- This report goes in the commit after that.

**Scope.** This report covers Part 0 only. Part A was not started. All I values are `log det J` (nat) at the nominal truth θ₀, on the σ/24 design grid unless σ/120 is stated. B = 316.8921168493721 for every input.

**Sources.** Paths are relative to `code_8_3_stage2/stage2_pkg/evaluator/`.
- `[L X:n]` is line n of `round13/logs/gate_X.log`, and `[L sum:n]` is line n of `round13/logs/gate_summary.log`.
- `[J a.b]` is field `a.b` of `round13/results/gate13.json`.
- Code: `round13/eval13.py`, `round13/dp13.py` (copies of `round11/eval11.py` and `round11/dp11.py`; the full diffs are in `round13/results/diff_eval11_eval13.txt` and `diff_dp11_dp13.txt`), `round13/common13.py` (helpers), `round13/run13.py` (driver), `round13/inputs13.json`.
- The U design's scan records are `round13/results/gate_U_design/scan_edge_j0.json` and `scan_edge_j1.json`; its summary is `design_edge.json`; the J matrices are `J_*.npy`.
- Environment: Python 3.13.18, numpy 2.5.3, scipy 1.18.1, numba 0.68.0 (the same versions as Round 11, `round11/results/state_inputs.json`), 4 vCPU, 16 GB RAM.

## 1. Conclusion

**All five gates G0–G4 pass, so Part A may start.** For input U, the round-13 copies reproduce Round 11 bit for bit: the module grid, all 24 DP runs of the two scans, the codes of T₀ and T₁, their I and E[N] on σ/24 and σ/120, and the certificate UB = 207.56048558735236. The six inputs register with the identified G and I_full.
Sources: [L sum:2–7]; [J gates.\*, all_pass].

## 2. Gates, one line each

| gate | what was compared | worst deviation | result |
|---|---|---|---|
| G0(a) | `eval13` against `eval11` in one process: XK, HK, Z, KA, KB, FL, CL (dtype, shape, bytes), plus DZ, Z0, G, X_PAD, NAME, B, s | none (all bitwise equal) | PASS [L U:4] |
| G0(b) | `design(edge_set(), gamma_from_J, logdet9, 9, 'edge')` against `round11/results/scan_edge_j0.json` and `scan_edge_j1.json`, run by run: lam, EN, S, F, dsum, dual, logdet, feasible, phase; plus gbar, gamma1, final_run, min_dual, lam_bracket, run count | none: 12/12 and 12/12 runs bitwise; final_run 10 and 11; gbar/gamma1 0.021587304272182303/0.02987366782571452 (j0), 0.006849709430879255/0.009928291424921748 (j1) | PASS [L U:8, 21, 23, 36, 40–41] |
| G0(c) | codes of T₀ and T₁ against `round11/results/table_edge_A0.npz` and `table_edge_iter.npz` (4999 × 35146 uint16); I, EN24, I120, EN120 against `round11.json → tables.edge_j0 / edge_j1` (tolerance 1e-12 relative) | codes bitwise equal; relative deviation 0 for all 8 numbers (T₀: 207.10197960571716 / 316.61009947208913 / 207.0999194594685 / 316.413413652473; T₁: 207.35521730257756 / 316.7261244018059 / 207.35308449872582 / 316.5159931346839) | PASS [L U:42–43] |
| G0(d) | UB = I(T₀) − 9 + min dual(j1), recomputed, against the value from the archived fields | 207.56048558735236 against 207.56048558735236, difference 0 | PASS [L U:44] |
| G1 | `eval13.XK, HK` under R13_INPUT = P5514 against `dr.input_arrays('P5514')` (registered by `evaluator_window`) and `dr.input_arrays('D')` (config.json); under R13_INPUT = N against `dr.input_arrays('N')` | none (bitwise) | PASS [L P5514:4], [L N:4] |
| G2 σ/24 | `eval13.evaluate(ConstRows(480))` against the accepted operator (`des.sod_reference()` → `evaluator_fast`) and against the prompt's numbers (tolerance 1e-9 relative) | E[N] 316.8921168493916 against 316.8921168493721 (6.2e-14 rel); logdet 197.20097668224133 against 197.20097668224076 (2.9e-15 rel) | PASS [L U:6] |
| G2 σ/120 | `eval13.evaluate_refined(ConstRows(480), 5)` against `round7/results/archive_match.jsonl` entry (θ₀, M = 480, f = 5), which is present (tolerance 1e-9 relative) | E[N] 316.8911511746132 against 316.8911511745014 (3.5e-13 rel); logdet 197.20115853002983 against 197.2011585300231 (3.4e-14 rel) | PASS [L U:7] |
| G3 | LO: `fine_grid(5)` raises no BoundaryNode; `zf[0] == Z0`; every fine node's coarse cell is in [0, G−1]; ConstRows(480) on σ/24 and σ/120 with mass conservation ≤ 1e-12 / ≤ 1e-10 | zf[0] = Z0 = 1.8994231576096559 (U's default origin would be 1.9098150805683891); cells [0, 40014] = [0, G−1]; smallest distance to a cell boundary 0.1 cell widths; mass deviation 8.0e-14 (σ/24) and 2.5e-13 (σ/120) | PASS [L LO:4–5] |
| G4 | six inputs: Σa², output range, largest per-step motion, G, I_full; I_full to 1e-4, G exactly, step ≤ 4.6σ, abs(Σa² − 1) ≤ 1e-12 | largest abs(I_full − ref) 2.2e-5 (P5514); G equal for all six; largest step 4.28σ (LO) | PASS [L X:3] |

G3 numbers for LO with ConstRows(480): σ/24 E[N] = 367.814772611902, logdet = 197.35269715259415; σ/120 E[N] = 367.8138573050662, logdet = 197.3528664358578. The LO process also ran the accepted operator at Δ\* as a diagnostic: E[N] 367.814772611879 and logdet 197.35269715259363, i.e. 6.3e-14 / 2.6e-15 relative from eval13 [L LO:7].

G4 table [J g4_table.\*] (I_full in nat; range and step in σ; the reference values are the README's identification values):

| input | Σa² − 1 | output range | largest step | G | I_full | ref I_full | abs(diff) | G = ref |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| U | 0 | 1144.356 | 3.988 | 35146 | 219.008506 | 219.0085 | 5.9e-6 | yes |
| P5514 | −1.1e-16 | 1136.209 | 3.176 | 34951 | 219.007978 | 219.0080 | 2.2e-5 | yes |
| SCH | 0 | 1121.426 | 3.028 | 34596 | 219.008506 | 219.0085 | 5.9e-6 | yes |
| LO | −1.1e-16 | 1347.243 | 4.282 | 40015 | 218.034288 | 218.0343 | 1.2e-5 | yes |
| HI | −1.1e-16 | 869.106 | 3.436 | 28540 | 218.034288 | 218.0343 | 1.2e-5 | yes |
| N | 0 | 1482.514 | 4.039 | 43262 | 218.416907 | 218.4169 | 6.7e-6 | yes |

Range and step also agree with the README's printed values (1 and 2 decimals) for all six inputs. SCH phases: (0, 4.71238898038469, 1.5707963267948966, 3.141592653589793) = (0, 3π/2, π/2, π).

The U design, which is the G0(b)–(d) run, is identical to Round 11 and is also Part A's U entry. T\* = T₁ (edge_j1); I = 207.35521730257756 (σ/24) and 207.35308449872582 (σ/120); E[N] on σ/120 is 316.5159931346839 ≤ B. UB′ (B + 0.25) = 207.56525182437946, so the gap is UB′ − I(T\*) = **0.2100 nat** (≤ 0.5, no third scan). The forward/backward check gives 5.2e-13 (T₀) and 4.3e-13 (T₁); I(T) ≤ UB′ holds for both tables.
Sources: [L U:22, 37–39].

## 3. Wall times and memory

All runs used 4 numba threads and 1 BLAS/OMP thread. U and LO each ran alone: the four short gates (SCH, P5514, N, HI) had finished, and LO finished at 12:18:20 before U started at 12:18:28.

| | U (G = 35146) | LO (G = 40015) |
|---|---:|---|
| one DP run (edge set, 100 actions) | median 41.2 s over 24 runs (range 37.6–49.5 s) | 42.9 s (one run, λ = 5γ̄, tangent at LO's SoD) |
| one σ/24 forward, DP table | median 19.2 s over 24 runs (in the scans); 28.4 / 29.3 s for T₀ / T₁ with shape statistics | 22.5 s |
| one σ/24 forward, constant SoD (ConstRows(480)) | 4.6 s | 5.1 s |
| one σ/120 forward, DP table | 105.8 s (T₀), 110.2 s (T₁); G_f = 175724 | not run |
| one σ/120 forward, constant SoD | 24.9 s | 26.8 s (G_f = 200071) |
| fixed-policy backward, σ/24 | 15.7 / 17.4 s (T₀ / T₁) | not run |
| one scan (12 DP runs + 12 forwards) | 710 s (j0), 765 s (j1) | — |
| whole design (two scans, T₀/T₁ evaluations) | 1820 s; the gate process took 1856 s | — |

Sources: [J per_input.U.timing, per_input.LO.timing, per_input.LO.G3]; [L U:22, 37]; [L LO:5, 8].

**Peak memory of the U gate process: 9.1 GB** (VmHWM 9 095 468 kB, polled every 5 s from outside the process; `round13/logs/peak_rss_U_gate.txt`). The prompt plans for about 1.5 GB per process (see §4, item 1). A separate probe on T₁ (`round13/mem_probe.py`, output in `round13/logs/mem_probe.log`) measured three peaks:
- one σ/24 forward and one σ/120 forward without `Stats`: 0.55 GB;
- one σ/24 forward with `eval13.Stats`: 2.7 GB (1.34 × 10⁸ node values collected);
- after `Stats.result()`: 7.9 GB.

## 4. Where the prompt and the code at 1b88f01 differ (reported, not worked around)

1. **Memory per process.** The README and Part 0 §2.4 say each process holds about 1.5 GB, so four can run in parallel. The measured peak of the U gate process is 9.1 GB (§3). The probe in §3 puts the peak in `eval13.Stats`, copied unchanged from `round11/eval11.py`: 0.55 GB without it, 2.7 GB after the forward that fills it, 7.9 GB after `Stats.result()`. `Stats` keeps the per-node facing-edge offsets and masses of all 4999 steps (1.34 × 10⁸ values for T₁ of U, about 1.1 GB per float64 array); `result()` concatenates, sorts and accumulates them. `design()` runs `Stats` once per scan table, as Round 11's `post11.analyse` did. On this 16 GB machine, four such processes in parallel do not fit. Part 0 changes nothing here. For Part A this needs a decision (yours): (a) run at most one design process at a time while `Stats` runs (or run inputs one after another); (b) compute `Stats` only for T\*; or (c) a leaner statistics routine, which would no longer be a copy of `eval11.Stats`.
2. **G1 holds by construction.** `eval13.XK, HK` are built by `ew.design_grid(NAME)` → `dr.input_arrays(NAME)`. Comparing them with `dr.input_arrays(NAME)` compares the function with itself. The independent part of G1 is P5514 against `dr.input_arrays('D')`. That comparison uses the config.json copy of the same amplitudes and phases, and it is bitwise equal.
3. **G2 σ/120 compares two code paths.** The archived entry (θ₀, M = 480, f = 5) was produced by round 7's *reuse* route: the accepted operator `evaluator_fast` at n = 120 with 68σ support windows (`"route": "reuse"`). `eval13.evaluate_refined` is the Round-11 full-grid general path with 88σ kernel windows. The agreement is 3.5e-13 relative.
4. **`inputs13.json` as given.** `PAR` has `"p": "filled by Part B …"`, a string. §2.2 item 1's rule "every entry that has `p` or `a`" would try to register it, so it is skipped until Part B (DECISION 13-2). `SCH` has only `chi_rule`, no `chi`, so the phases are computed from the rule (DECISION 13-3).
5. Runtime guide (not a contradiction). DP runs took 41–43 s (guide: 45–60 s). σ/24 forwards of DP tables took 19–22 s (guide: 20–25 s). Constant-SoD forwards take only about 5 s, and σ/120 of a DP table takes about 110 s.
6. "Importing round4 … may create empty folders": none were created in this session. After every process, `git status` on rounds 4–11, config.json, density_recursion_v1.py and evaluator_fast.py was clean [L X:last line].

Everything else the prompt says about the code was checked against the code and holds. That covers eval11 line 43 / 56 / 329, evaluator_window lines 56–58, `dual = g1 + dsum + lam (B − 1)`, X_PAD = 960, `_step_full` stopping on any out-of-range edge, and all archived reference values.

## 5. DECISION list

- **DECISION 13-1.** Besides changes (1)–(4) of §2.2, `eval13.py` adds `import json` (needed for change 1) and a provenance header to the docstring. `dp13.py` adds only the docstring header. Archived diffs: `round13/results/diff_eval11_eval13.txt` and `diff_dp11_dp13.txt`.
- **DECISION 13-2.** Registration rules. An entry already in `dr.CFG['inputs']` is not re-registered: U and N come from config.json, P5514 from evaluator_window. If such an entry lists numeric `a`/`chi` (U), they are asserted equal to the registered values. An entry is registered only if its `a` or `p` is a numeric 4-vector, so PAR is skipped in Part 0. Amplitudes are a = sqrt(p) in float64.
- **DECISION 13-3.** SCH phases are computed from `chi_rule`: `np.mod(-π m (m−1)/4, 2π)`, m = 1..4, float64. They are asserted equal to (0, 3π/2, π/2, π) within 1e-15 absolute.
- **DECISION 13-4.** The check abs(Σa² − 1) ≤ 1e-12 is applied to every input in `dr.CFG['inputs']` (U, D, E, N, P5514, SCH, LO, HI), not only to the newly registered ones.
- **DECISION 13-5.** The KA/KB range assertion of §2.2 is placed in `eval13` at module level, after the grid, and is also kept in `Design11.__init__`.
- **DECISION 13-6.** `matched_sod` tie rules. On σ/24, a tie in abs(E[N] − B) between the two bracketing M picks the larger M (the σ/120 rule of the prompt). The σ/24 record of M\* is evaluated if it was not visited. The bisection keeps the bracket by sign of E[N] − B.
- **DECISION 13-7.** `design()` evaluates every scan's final table (T₀, T₁, and T₂ if run) on σ/24 and σ/120, runs the fixed-policy backward check on each, and computes `Stats` for each with the γ of scan j0 (tangent at SoD; the convention of `round11/post11.analyse`). If the criterion ties, T\* is the earlier table. After writing its json, the run stops if T\*'s σ/120 E[N] > B, if a table's criterion exceeds UB′, or if the forward/backward check exceeds 1e-12.
- **DECISION 13-8.** Scan checkpoint. A scan is skipped only if its state json has a final run, its tangent (gbar, γ₁) equals the recomputed one bitwise, and the final table's `.npy` still exists. Otherwise it is re-run.
- **DECISION 13-9.** DP tables go in `$ROUND13_WORK/<input>/<label>/`. Labels repeat across inputs, so the work directory is split by input. In this session `ROUND13_WORK` was the session scratchpad (not committed), so the `table` paths in the scan jsons point there.
- **DECISION 13-10.** The `gate` stage runs per input in its own process (U: G0, G2, G4; P5514 and N: G1, G4; LO: G3, DP timing, G4; SCH and HI: G4). A separate stage `gate_summary` collects the six files into `gate13.json`. The `partA` and `partB` stages refuse to run in this commit.
- **DECISION 13-11.** Threads: 4 numba threads per process (`R13_NUMBA_THREADS`, as in Round 11's driver); OMP/BLAS 1.
- **DECISION 13-12.** G4 pass rule: abs(I_full − ref) ≤ 1e-4, G equal, largest step ≤ 4.6σ, abs(Σa² − 1) ≤ 1e-12. Range and step are also compared with the printed reference values (1 and 2 decimals) as information.
- **DECISION 13-13.** G2 σ/24 is compared both with `des.sod_reference()` and with the prompt's numbers (316.8921168493721, 197.20097668224076). Both comparisons pass.
- **DECISION 13-14.** LO's DP timing run uses the edge set with tangent at LO's SoD and λ = 5γ̄ (the first run of a scan). Its table is also evaluated on σ/24 as a diagnostic: E[N] 82.79, I 197.30, DP value against policy 2.9e-13 [L LO:8].
- **DECISION 13-15.** `ConstRows.rows_full` returns the same two arrays every step (no copy). The mass check on clipped nodes runs only for M > 958, with the strict inequalities of §2.4.
- **DECISION 13-16.** Extra checks in G1 (reported, part of the pass flag): P5514's a and χ equal `round2a/config_recheck.json` candidates[0], and round 13 does not re-register P5514 or N.

## 6. Files

`round13/`: `inputs13.json`, `eval13.py`, `dp13.py`, `common13.py`, `run13.py`, `.gitignore` (`work/`, `__pycache__/`); `results/gate13.json`, `results/gate_<input>.json` (six), `results/gate_U_design/` (two scan jsons, `design_edge.json`, five `J_*.npy`), `results/diff_*.txt`; `logs/gate_<input>.log` (six), `logs/gate_summary.log`, `logs/peak_rss_U_gate.txt`; `mem_probe.py` and `logs/mem_probe.log` (memory probe of §3). No DP table is committed: the U tables are bitwise equal to the packed Round-11 tables.

Commands (from `round13/`, with `OMP_NUM_THREADS=1`):
```bash
R13_INPUT=U ROUND13_WORK=<dir> python3 run13.py gate
```
The same command is run for SCH, P5514, N, HI and LO, then `python3 run13.py gate_summary`.

## 7. Not done

- Part A and Part B, as instructed. No predictions file was written; Part 0 has none.
- `matched_sod` was written but not run: no Part 0 gate requires it. It will run for the first time in Part A. `ConstRows` with M > 958 (clipping) was not exercised either.
- No σ/120 evaluation of a DP table for LO. LO's memory was not measured.
- PAR is not registered; Part B fills it.
- The memory issue of §4 item 1 is reported and not changed. It needs a decision before Part A runs inputs in parallel.
