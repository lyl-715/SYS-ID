# Round 13 — Part 0: infrastructure for other inputs, and regression gates

Read `README_round13.md` first (purpose of the round, order of the three parts). This part builds the plumbing that Parts A and B use and proves that it changes nothing for the nominal input. **No result of Part A or B counts unless every gate of this part passes.**

## 0. Rules

- Repository `lyl-715/SYS-ID`. Create branch `round13-input-trigger` from `round11-edge-actions` at `1b88f01`. Push only to this branch. `claude/optimistic-newton-xv5jvl` stays at `3314e7d`, `round10-upper-bound` at `2b1af1b`, `round11-edge-actions` at `1b88f01`.
- Everything under `code_8_3_stage2/stage2_pkg/evaluator/round4 … round11` and `evaluator/config.json`, `density_recursion_v1.py`, `evaluator_fast.py` is read-only. New code only in `evaluator/round13/`.
- First commit: the four `.md` files of this package into `reports_CN/`.
- Importing `round4/trigger_dp.py` or `round4/evaluator_window.py` may create empty folders under `round4/`; delete them after each run so `git status` on rounds 4–11 stays clean.
- Assertion failures (mass conservation, non-negativity, support boundary, any gate below) stop the run. They are not fallbacks.
- Every choice this prompt leaves open is written in the report as `DECISION 13-x: …`.
- Work directory for DP tables: `$ROUND13_WORK` (default `round13/work/`, not committed). Commit only packed final tables (`np.savez_compressed`, as `round11/post11.pack`).

## 1. Why new modules are needed (verified in the code at `1b88f01`)

`round11/eval11.py` binds the scenario at import time to the config input `'U'`:

```
NAME = 'U'
XK, HK, Z, DZ, KA, KB = ew.design_grid(NAME)          # line 56
```
and `round11/dp11.py` copies those module globals (`Z, DZ, XK, HK, KA, KB, FL, CL, Z0`). A different input therefore needs its own process with these globals rebuilt. Two more places are tied to `'U'`:

1. `eval11.fine_grid` calls `er.cell_index(zf)` from `round7/evaluator_refine.py`, whose defaults `z0=None, h=None` are **the coarse origin and step of input `'U'`**. For another input the coarse origin differs, so the call must pass that input's own origin and step.
2. `dr.input_arrays(name)` reads `dr.CFG['inputs'][name]`; only `U, D, E, N` are in `config.json`. `round4/evaluator_window.py` already registers a fifth input in memory (`dr.CFG['inputs']['P5514'] = dict(a=…, chi=…)`, lines 56–58) — the same mechanism is to be used for the new inputs. `config.json` is not edited.

## 2. Files to create in `evaluator/round13/`

### 2.1 `inputs13.json`
```
{
 "U":     {"a": [0.5, 0.5, 0.5, 0.5], "chi": [0, 0, 0, 0], "source": "config.json inputs.U"},
 "P5514": {"source": "round2a/config_recheck.json candidates[0] (already registered by evaluator_window)"},
 "N":     {"source": "config.json inputs.N"},
 "SCH":   {"p": [0.25, 0.25, 0.25, 0.25], "chi_rule": "chi_m = (-pi m (m-1)/4) mod 2pi, m = 1..4  ->  (0, 3pi/2, pi/2, pi)"},
 "LO":    {"p": [0.40, 0.30, 0.20, 0.10], "chi": [0, 0, 0, 0]},
 "HI":    {"p": [0.10, 0.20, 0.30, 0.40], "chi": [0, 0, 0, 0]},
 "PAR":   {"p": "filled by Part B (classical parametric-optimal input)", "chi": [0, 0, 0, 0]}
}
```
`p_m = a_m^2`; amplitudes are `a = sqrt(p)` computed in double precision at registration. For `U`, `P5514`, `N` the amplitudes and phases are taken from the cited sources at full precision, not retyped. All inputs keep `a0 = 1` (`dr.a0`) and the four frequencies `dr.w`.

### 2.2 `eval13.py` — a copy of `round11/eval11.py` with exactly these changes
1. After `dr = ew.dr`: load `inputs13.json` and, for every entry that has `p` or `a` and is not already in `dr.CFG['inputs']`, register `dr.CFG['inputs'][name] = dict(a=[…], chi=[…])` (in memory only). Assert for every registered input `abs(sum(a_m^2) - 1) <= 1e-12`.
2. `NAME = os.environ.get('R13_INPUT', 'U')`; assert `NAME in dr.CFG['inputs']`.
3. In `fine_grid`: `cell, gap = er.cell_index(zf, z0=Z0, h=DZ)`.
4. Nothing else. In particular `B = ew.B_BUDGET` (= 316.8921168493721) stays the budget for every input, and `s`, `D`, `K`, the grid rule `dr.make_grid(xk, 24, 8*D, 'B')` and the kernel windows `ew.windows88` are unchanged.

After the module-level grid is built, add two input-specific assertions (they hold for `U` and must hold for every input, since the support argument of the paper assumes them):
- `max_k |XK[k+1] - XK[k]| <= 4.6 * s` (the per-step output motion bound; values for the inputs above are 3.0–4.3 σ);
- `int((KA[1:] - ew.X24).min()) >= 0 and int((KB[1:] + ew.X24).max()) <= G - 1` (already asserted by the DP constructor; keep it there).

### 2.3 `dp13.py` — a copy of `round11/dp11.py` with one change
`import eval13 as e11` instead of `import eval11 as e11`. Nothing else.

### 2.4 `common13.py` — new helpers (used by Parts A and B)
- `full_information()` → `J_full = HK.T @ HK / s**2` and `I_full = slogdet(J_full)`.
- `class ConstRows` with attribute `M` (assert `48 <= M <= 1920`) and method `rows_full(p)` returning two `int64` arrays over the full grid: `L_i = min(M, i + X_PAD − 2)`, `U_i = min(M, G − 1 − i + X_PAD − 2)`. It lets `eval13.evaluate` and `eval13.evaluate_refined` evaluate a constant-threshold SoD of half-width `M` grid units (threshold `M*s/24`) on the full grid. The clipping is needed because `eval11._step_full` stops when *any* node's window edge leaves the padded index range `[−X_PAD, G−1+X_PAD]`, including nodes that carry no mass; the `− 2` keeps the refined evaluation (which multiplies half-widths by 5 at fine nodes offset by up to two fine steps from the coarse node, on a fine grid whose last node may fall a few fine steps short of the coarse grid's end) inside `5·X_PAD`. For `M ≤ X_PAD − 2 = 958` nothing is clipped. Clipped nodes must never carry mass: after every forward with `M > 958` assert `fw['sup_lo'].min() > M − X_PAD + 2` and `fw['sup_hi'].max() < G − 1 − (M − X_PAD + 2)` (scaled by 5 on σ/120); stop if violated.
- Every reported evaluation (constant SoD at Δ*, matched SoD, `T0`, `T1`, `T*`, on σ/24 and σ/120) saves its 9×9 `J` as `.npy` next to the json (Part B reuses them).
- `matched_sod(B)`: the constant SoD whose expected number of transmissions is closest to `B`.
  1. σ/24: evaluate `E[N](M)` with `eval13.evaluate(ConstRows(M))` starting at `M = 480` (= Δ*), stepping `M` by ±20 in the direction that moves `E[N]` towards `B` until `B` is bracketed, then bisect on the integer `M` until two consecutive integers bracket `B`. Let `M̂` be the one with the smaller `|E[N] − B|`.
  2. σ/120: `eval13.evaluate_refined(ConstRows(M))` for `M ∈ {M̂−1, M̂, M̂+1}`; if the minimiser of `|E[N]_{120} − B|` is at an end of this set, extend the set by one on that side and repeat. `M*` = the minimiser (ties → larger `M`).
  3. Accept if `|E[N]_{120}(M*) − B| <= 1.0` (round 7's D0 rule); report the signed residual. Return `M*`, `E[N]`, `S`, `J`, `logdet` on σ/24 and σ/120.
- `design(acts, gamma_fn, crit_fn, p_dim, label)`: the two-scan design used in both parts. `crit_fn(J)` is the criterion (Part A: `slogdet(J)[1]`, `p_dim = 9`; Part B: `slogdet(Dᵀ J D)[1]`, `p_dim = 2`). The tangent matrices are formed exactly as the `edge` stage of `round11/run_round11.py` forms them (same evaluation of the table, same function for `gamma`); gate G0(b) checks this bitwise through `gbar` and `gamma1`.
  1. `rec_sod, J_sod, W_sod = des.sod_reference()` (constant SoD at Δ* for this input, accepted operator).
  2. Scan `j0`: tangent at `J_sod`, i.e. `gam, gbar, g1 = gamma_fn(J_sod)`; `Scan11(des, acts, gam, gbar, g1, f"{label}_j0", work, log).scan()`; `T0` = the scan's final table; `J(T0)` from `eval13.evaluate`.
  3. Scan `j1`: tangent at `J(T0)`; `T1` = final table.
  4. Best table `T*` = whichever of `T0, T1` has the larger `crit_fn` value on σ/24.
  5. Certificate from scan `j1` (tangent point `T0`): for budget `B`, `UB = crit(T0) − p_dim + min_λ dual(λ)`; for `B' = B + 0.25`, `UB' = crit(T0) − p_dim + min_λ [dual(λ) + 0.25 λ]`, where `dual` is the field written by `Scan11.run` (`g1 + dsum + lam*(B − 1)`). Report `UB`, `UB'`, `gap = UB' − crit(T*)` (σ/24), and the argmin λ. If `gap > 0.5` nat, run a third scan `j2` with tangent at `T*`, recompute the certificate from it, and take the better table and the smaller bound.
  6. σ/120 re-evaluation of `T*`: `E[N]_{120} <= B` is required for a reported table; report `crit_{120}`, `S_{120}`.
  7. Shape statistics of `T*` with `eval13.Stats` (facing-edge 10/50/90 % quantiles; per-bin transmissions and information).
  `gamma_fn` for Part A is `Design11.gamma_from_J` (unchanged). Scans are checkpointed: after each completed scan write its state json; a re-run skips completed scans.
- One driver `run13.py` with stages `gate`, `partA`, `partB`; each (input, stage) runs as a separate process: `R13_INPUT=<name> ROUND13_WORK=… python3 run13.py <stage>`. Up to four processes in parallel (each holds about 1.5 GB); `OMP_NUM_THREADS=1` as in the earlier rounds.

## 3. Gates (all blocking)

**G0 — the copies are faithful (input U).** With `R13_INPUT=U`:
- (a) `eval13.XK, HK, Z, KA, KB, FL, CL` are bitwise equal to those of `eval11` (compare in one process that imports both).
- (b) `design(edge_set(), Design11.gamma_from_J, logdet9, 9, 'edge')` (with `logdet9(J) = slogdet(J)[1]`) reproduces `round11/results/scan_edge_j0.json` and `scan_edge_j1.json` run by run: `lam`, `EN`, `S`, `F`, `dsum`, `dual`, `logdet`, `feasible` bitwise for every run, the same `final_run` (10 and 11), `gbar` and `gamma1` bitwise (j0: 0.021587304272182303 / 0.02987366782571452; j1: 0.006849709430879255 / 0.009928291424921748).
- (c) the codes of `T0`, `T1` equal `round11/results/table_edge_A0.npz` and `table_edge_iter.npz` bit for bit; `I`, `EN24`, `I120`, `EN120` equal `round11.json → tables.edge_j0 / edge_j1` to 1e-12 relative (for identification: 207.10197960571716 / 316.61009947208913 / EN120 316.413413652473; 207.35521730257756 / 316.7261244018059 / EN120 316.5159931346839).
- (d) the certificate from scan `j1`: `UB = I(T0) − 9 + min_dual(j1) = 207.10197960571716 − 9 + 9.458505981635199 = 207.560485587352…`; must agree to 1e-12 with the value recomputed from the archived fields.

**G1 — input registration.** For `P5514`: `XK`, `HK` built through `eval13` are bitwise equal to `dr.input_arrays('P5514')` as registered by `evaluator_window` and to `dr.input_arrays('D')` (config). For `N`: bitwise equal to `dr.input_arrays('N')`.

**G2 — constant SoD on the full grid.** For `U`: `eval13.evaluate(ConstRows(480))` against the accepted operator (`des.sod_reference()`): `E[N]` and `logdet` agree to 1e-9 relative (reference 316.8921168493721 and 197.20097668224076). On σ/120, compare with `round7/results/archive_match.jsonl` entry `(theta_0, M = 480, f = 5)` if present (1e-9 relative); if absent, report the value and say so.

**G3 — refined grid for a non-U input.** For `LO`: `fine_grid(5)` must run without `BoundaryNode`, `float(zf[0]) == Z0`, and every fine node's coarse cell index lies in `[0, G−1]`. Evaluate `ConstRows(480)` on σ/24 and σ/120 and report `E[N]`, `logdet`, mass-conservation deviation (≤ 1e-12 and ≤ 1e-10).

**G4 — sanity of the inputs.** For each of `U, P5514, SCH, LO, HI, N`: report `sum a^2`, output range in σ, largest per-step motion in σ, `G`, and `I_full`. For identification (computed independently before this prompt was written): `I_full` = 219.0085 (U), 219.0080 (P5514), 219.0085 (SCH), 218.0343 (LO), 218.0343 (HI), 218.4169 (N); largest per-step motion 3.99, 3.18, 3.03, 4.28, 3.44, 4.04 σ; `G` = 35146, 34951, 34596, 40015, 28540, 43262. Agreement of `I_full` to 1e-4 and of `G` exactly is required.

## 4. Output of Part 0

- `round13/results/gate13.json` with every compared number and a pass flag per gate.
- `reports_CN/report_round13_part0_CN.md` (+ EN copy): one line per gate (what was compared, worst deviation, pass/fail), the DECISION list, wall times of one DP run and one σ/24 / σ/120 forward for `U` and for `LO`.
- If any gate fails: stop, commit the log and the report, do not start Part A.
