# §8.5 stage 4: design (frozen before any stage-4 run; 2026-10-13)

This file fixes every rule of Scripts 1–3 that the stage-4 prompt leaves open. It was committed together with `predictions.md`, before any stage-4 script ran. Where the scripts make a choice not listed here, the report lists it.

## 0. Common

- **Working directory and outputs.** Work in `code_8_3_stage2/stage2_pkg/`. Outputs go to `stage4/`; `stage2/` and `stage3/` are only read.
- **Reused code, unchanged.** `stage3_common.py` exec's `stage2_common.py` and the Nelder–Mead block of `stage2_local.py`. Reused from them:
  - setup, `sod_batch` and `check_trigger` (bitwise proof against `mc83.sod`);
  - `evaluate_crn`, `summarize_s2`, `pareto_front`, `eval_many` and `diff_se`;
  - `x_to_input`, `input_to_x`, `eval_s`, `pen` and `nm_run`.

  Every script runs `check_trigger()` before any evaluation.
- **Budget decisions.** `mc85_post.py::qual` (8.3.6(g), via `stage2_common.py`) is used unchanged for screening, for the R = 2000 ranking and front, and for Script 3. Script 2 uses its own classifier (§2).
- **Space.** The full 7-dimensional space throughout: p = softmax(z1, z2, z3, 0), a_m = √p_m, and four continuous phases. There is no phase reduction.
- **Identity of a point.** `key_of(p, χ)`: p and χ mod 2π, each rounded to 7 decimals, as in stage 3. Deduplication always uses this key, never the id.
- **Feasibility at R = 2000.** For floor λ0 a candidate is feasible if `qual = 'ok'` and m_E^(C) ≥ λ0 − 1e-9, as in stages 2–3. For λ0 = none only the budget applies.
- **Draws.**

  | draws | seed | R | use |
  |---|---|---|---|
  | search | 20261010 | 500 | inside NM only, the stage-2 search draws |
  | ranking | 20261012 | 2000 | every Script-1 ranking, start and endpoint, merged with the stage-2/3 pool |
  | confirmation | 20261020 | 8000 | Script 2 only |
  | scenarios | 20261021 | 4000 | Script 3 only |
  | start selection | 20261018 | — | random choice of Script-1 starts; no noise |

## 1. Script 1 (`stage4_search.py`)

### 1.1 Previously used starts

A point counts as "used as a start in stages 2–3" if its key equals the key of x0 of any Nelder–Mead task in `stage2/local_runs.json`, `stage3/dense_runs.json` or `stage3/unconstrained.json` (40 tasks, 27 distinct keys). Sweep and grid centres of `stage3_peak.py` and `stage3_dense.py` are not NM starts, so they are not excluded (e.g. NM-none-P5514 and AB-0.5).

### 1.2 Stage-2/3 pool and current front

- **Stage-2/3 pool.** All R = 2000, seed-20261012 records: `stage2/pool_R2000.json` (54), `stage3/dense_R2000.json` (56) and `stage3/unconstrained.json` (10 U- endpoints, all over budget). 120 records.
- **Current front.** The script recomputes the front of the qualified records with `pareto_front` and asserts that it equals `stage3/front_union_R2000.json`: 16 points.

### 1.3 Front starts (6 per λ0)

- **Eligible:** the current front points that were not used as starts (§1.1). There are 11.
- **m_E strata, on R = 2000 m_E^(C):**

  | stratum | m_E range | eligible points (m_E) |
  |---|---|---|
  | L | [0, 0.1) | S3-none-G-P5514(0,0,−30,−30) (0.045), NM-none-P5514 (0.045), S3-none-PO-NM-none-P5514 (0.046), S3-0.3-NM-0.3-P5014 (0.047) |
  | M | [0.1, 0.5) | S3-none-NM-none-P5450 (0.135), P1045 (0.141), S3-0.3-NM-0.3-P4312 (0.423), S3-0.3-NM-0.7-P4312 (0.440) |
  | H | [0.5, ∞) | AB-0.5 (0.531), G-AB05(30,30,60,60) (0.605), S3-0.7-P4055 (0.705) |

  Excluded as used: PO-NM-none-P5514, P1001, P4312, NM-0.3-P4312 and UNIF.
- **Request:** 2 per stratum per λ0.
- **Generator:** `rng = np.random.default_rng(20261018)`, consumed in this order:
  1. `rng.permutation(n_L)`, `rng.permutation(n_M)` and `rng.permutation(n_H)`, each applied to the stratum's eligible points sorted by id;
  2. the R = 200 sample of §1.4.
- **Sampling without replacement, by cyclic dealing.** Each stratum's permutation is dealt to the slots in the order λ0 = none, 0.3, 0.5, 0.7, with 2 slots per λ0.
  - A pointer walks through the permutation cyclically.
  - A point already dealt to the same λ0 is skipped, and the next one is taken.
  - So within a λ0 the starts are distinct. Across λ0 a stratum is used without replacement until it is exhausted, then the cycle restarts. Re-use across λ0 is unavoidable: 11 eligible points serve 24 slots.
- **A stratum with fewer than 2 eligible points:** all of its points are used for every λ0. The shortfall is requested from the nearest other stratum in the order L–M–H, with ties going to the higher-m_E stratum.
- **Fewer than 6 eligible front points overall:** the λ0 gets fewer front starts, and this is logged. No previously used start is added, and the R = 200 part is not enlarged.

  With the current counts (4/4/3) neither rule is triggered.

### 1.4 R = 200 pool starts (6 per λ0)

- **Eligible:** records of `stage2/merged_R200.json` (stage-1 + LHS, R = 200, seed 20261010) with `dec == 'ok'`, minus used starts (§1.1) and minus the Script-1 front starts. There are 2729 or slightly fewer. Stage-3 grid points are not part of this pool.
- **Sampling:** sorted by id, then `rng.choice(n, 24, replace=False)` with the same generator. So all 24 are distinct across λ0. The first 6 drawn go to λ0 = none, then 6 each to 0.3, 0.5 and 0.7.

### 1.5 Runs

- **Search settings.** Budgeted runs use `nm_run` unchanged, with stage-2 penalty weights MU_B = 1 nat/send on max(0, D + 2se) and MU_E = 20, simplex steps 0.25/0.25, and MAXFEV = 300. The inner draws are R = 500, seed 20261010, with the uniform reference on them (`setup_search`).
- **Unbudgeted problem.** 10 starts: the 10 highest-Δm_D^(C) records of `stage2/merged_R200.json` (any decision) whose key was not used as a start in stages 2–3. These are ranks 11–20 of the ranking stage 3 used: L0299, L1573, L2029, L0295, L2207, L2375, L1870, L0190, L2916 and L2066. The R = 2000 U- endpoints of stage 3 are not used as starts.
  - The search is `nm_run` with MU_B = 0 and λ0 = none, as in `stage3_unconstrained.py`, with MAXFEV = 300.

### 1.6 Archive rule (two kept points per run)

For each run:
1. **Endpoint:** `x` returned by `minimize`, the best simplex vertex. Its origin is the run and the evaluation index (1-based) of the identical archived x.
2. **Constrained best:** among the run's archived evaluations that satisfy the run's own constraints on the search draws (R = 500 values in the archive), the one with the largest Δm_D. Ties go to the earliest evaluation.
   - *Constraints:* budget D + 2·Dse < 0, which is the `qual` 'ok' rule applied to the R = 500 values. Plus m_E ≥ λ0 if the run has a floor.
   - *Unbudgeted runs* have no constraint, so this is the archive maximum of Δm_D.
   - *If a run has no archived point satisfying its own constraints on the search draws, keep only the endpoint and log that fact; do not relax the constraint ad hoc.*

**Provenance and deduplication.**
- Recorded per kept point:
  - `kept_as`: list of (run, 'endpoint' | 'constrained best', evaluation index);
  - `origin_run`: every Script-1 run whose archive contains exactly this x, with its evaluation index, as in stages 2–3.
- Deduplication is by key over all kept points.
- A kept point whose key is already in the stage-2/3 pool is not re-evaluated. Its existing R = 2000 record is used: same draws, so bitwise identical. This is logged.

### 1.7 Re-evaluation, merge, front

**Starts at R = 2000.** Every start (48 budgeted + 10 unbudgeted, deduplicated) is evaluated at R = 2000, seed 20261012.
- For starts that already have a stage-2/3 record, the script asserts bitwise equality of the record's numbers and keeps the existing record.
- New start records (the R = 200 pool starts) enter the merged pool, with src `stage4-start`.

**Kept points.** New kept points are evaluated at R = 2000 with `summarize_s2`. The merged pool is the stage-2/3 pool plus all new stage-4 records. The front is `pareto_front` over the qualified records.

### 1.8 Per-floor report

- **Previous best:** the best feasible record in the stage-2/3 pool.
- **New best:** the best feasible record in the merged pool.
- **Paired difference:** new − previous, with se from `diff_se` (10 batches × 200, same draws).
- **Exploratory interval:** difference ± t_{9, 0.975}·se, with t = 2.262.
- **Threshold 0.02 nat:** both flags are reported, "point estimate above 0.02" and "interval lower end above 0.02".
- **Statement (i):** *Intervals in Script 1 are exploratory diagnostics: the same R=2000 draws are used for selection and ranking, so 'interval lower end above 0.02' there does not carry the approximate 95% overall confidence of Script 2.*
- **Statement (ii):** *If a run has no archived point satisfying its own constraints on the search draws, keep only the endpoint and log that fact; do not relax the constraint ad hoc.*
- **Start → endpoint.** A start is "feasible" for its run if it is feasible at R = 2000 for the run's floor (§0: budget and floor).
  - Feasible starts get (i) the Δm_D change, endpoint − start, with paired se, plus the endpoint's feasibility.
  - Infeasible starts get (ii) the transition (feasible / infeasible at the endpoint), with the failing constraint(s) at start and end.

  Both constraints are shown separately, so either reading of "satisfied the floor" can be recovered.
- **Unbudgeted problem.**
  - Previous best: the largest Δm_D^(C) over all stage-2/3 records (U-L1657).
  - New best: over the merged pool, any decision.
  - Paired difference, se and the two flags, as above.
- **Search budget.** NM evaluations per floor in stages 2, 3 and 4 are taken from the run files, and reported against the improvement found. No saturation claim is made.

## 2. Script 2 (`stage4_confirm.py`): frozen pool, one independent batch

- **Two invocations.**
  - `python3 stage4_confirm.py freeze` writes `stage4/confirm_frozen.json` and `stage4/stage4_confirm_freeze.log`, then exits without drawing noise. It records the pool, incumbents, δ, confidence, method, stopping rule, family, M and t.
  - These two files are committed.
  - `python3 stage4_confirm.py draw` refuses to run unless the frozen file is tracked by git and unmodified. It then draws seed 20261020.
- **Pool.**
  - All points of the Script-1 front (`stage4/front_R2000.json`), in front order.
  - Then the 7 stage-2 finalists (ids from `stage2/final_batch1.json`): NM-none-P5514, NM-0.3-P4312, AB-0.5, NM-0.7-UNIF, UNIF, NM-none-P5450 and P5014.
  - Deduplicated by key, with no truncation. UNIF is always in the pool (it is a finalist).
- **Incumbent per floor:** the Script-1 new best at R = 2000 (§1.8). It is a front point, so it is in the pool.
- **Evaluation.** `evaluate_crn(a, χ, Z, nbatch=20)`, unchanged: 20 batches of 400, chunk 200, R = 8000, with the uniform reference evaluated on the same Z. UNIF is also evaluated as a pool candidate, as a self-check: D = 0 and Δm_D = 0 exactly.
- **Statistics.**
  - *Point estimates* use the full R = 8000 matrices:
    - D = mean(cnt − cnt_u);
    - Δm_D^(C) = log det J − log det J_C;
    - m_E^(C) = λ_min(J, J_C);
    - pairwise Δm_D(i) − Δm_D(j).
  - *Batch statistics*, formed within batch b (400 draws):
    - D_b = batch mean of the paired differences cnt − cnt_u;
    - Δm_D,b = log det J_b − log det J_C,b;
    - m_E,b = λ_min(J_b, J_C,b);
    - pairwise (log det J_i,b − log det J_j,b).
  - *se* = sd(batch statistics, ddof 1)/√20.
  - *Interval:* point estimate ± t_{19, 1−0.05/(2M)}·se, with M = 2(n − 1) + C(n, 2).
- **Nonlinearity diagnostic (not a bias bound).** Mean of the batch statistics minus the full-sample statistic, for Δm_D and for m_E, per candidate.
- **Classifier.** As in the prompt; UNIF keeps its definitional values (budget confirmed, m_E = 1).
  - Feasibility at floor λ0 is **rejected** if the budget is rejected or (λ0 ≠ none and m_E ≥ λ0 is rejected).
  - It is **confirmed** if the budget is confirmed and (λ0 = none or m_E ≥ λ0 is confirmed).
  - Otherwise it is **undecided**.

  Challengers, exclusion with δ = 0.02 nat, and the two-group candidate near-optimal set follow the prompt. The floors are none, 0.3, 0.5 and 0.7.

## 3. Script 3 (`stage4_scenarios.py`): d = 5 and d = 50

**Representatives.** These are the four non-uniform floor bests after Script 1, at R = 2000 in the merged pool, plus UNIF.

**Replacement rule.**
- Floors are processed from the strictest to the least strict: 0.7, 0.5, 0.3, none.
- For each floor, the representative is the feasible (§0) non-uniform candidate with the largest Δm_D^(C) that is not already the representative of a stricter floor.
- So a candidate that is best at several floors represents the strictest of them, and each less strict floor takes its next-best feasible non-uniform candidate.
- If a floor's best is UNIF, the best non-uniform feasible candidate replaces it.
- If no non-uniform feasible candidate is left for a floor, that floor has no representative, and this is logged; there is no relaxation.

Script 1 applies the rule and saves the representatives. Script 3 re-derives them and asserts equality. The Script-2 classification does not change the representatives; their Script-2 status is reported next to the Script-3 results.

**Scenario rebinding.** For each d, σ_d = Δ*/d is rebound everywhere before any evaluation and before the worker pool forks:
- the module global `s`, which `evaluate_crn` uses for the noise (z = x + s·Z), for α and β in c_k (`cfun`) and for J = HᵀWH/s²;
- `Jb` and `ldB`, used by `summarize_s2` for Δm_D^(B) and m_E^(B), are first deleted. They are then rebuilt as J_per(u_unif, N_d) = H_pᵀH_p/σ_d², with H_p = `regressors(a_u, χ_u, t_i)`, t_i = iT/N_d, i = 1..N_d;
- `ti`, `Hp` and `Nper` are rebound likewise.

The script asserts that Δm_D^(B) from `summarize_s2` equals log det J_u − log det J_per computed locally. `check_trigger()` is re-run after each rebinding.

**Draws.** Seed 20261021, R = 4000, the same Z for d = 5 and d = 50. CRN is shared across d, so the two scenarios are not independent batches. Batch se uses `evaluate_crn`'s default 10 batches × 400.

**Uniform reference.** The scenario's own uniform reference on the same draws gives:
- B̂_d with se = sd/√R;
- N_d = ⌊B̂_d⌋;
- J_per rank, as the number of eigenvalues > 1e-9·max (the `mc83.py` convention), and condition number max/min eigenvalue.

**Budget decision:** `qual` (8.3.6(g)) applied to the paired D vs the scenario's reference.

**Addition: d = 20 control.** d = 20 is also evaluated on the same seed-20261021 draws, with N_20 = ⌊B̂_20⌋ from these draws. Then the direction of change d = 20 → 5 and d = 20 → 50 is a paired comparison. It is labelled as a control and is not one of the two scenarios.

**Window input power.** (1/K)Σ_k u_ac(t_k)², as in `evaluate_crn`, checked for exact equality with the representative's d = 20 record.
