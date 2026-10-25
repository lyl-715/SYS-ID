# Decisions — round 7 (spec `reports_CN/plan_round7_CN.md` v3.1, frozen)

Each line records a choice the spec (and the task) does not make, the reading taken (the most conservative one available)
and where it acts. Overnight mode: none of these stopped the run. Nothing was tuned to a target. Wording: 固定设计、离散方案下;
nothing here is a bound. Round 6's verdict (case 1) stands and is not revisited.

## Part 0 (round-6 fixes; one commit, no numerical change)

- DECISION 0-1: `round6/margin_design.py final` now recomputes every row of `round6/results/margin_final.json` from committed
  files only (the three margin npz tables, round 4's `table_U_W.npz`, the scan records, round 4's `scan_U_W.json` and
  `final_checks.json`; each checked with `git ls-files` and against HEAD) through the scan's own evaluation path
  (`evaluator_window.evaluate_table`) plus the round-4 forward-backward check, and compares field by field (exact; wall-clock
  fields excluded); it writes `results/margin_final_npz.json` and `logs/partA1_final_npz.log` and overwrites nothing. Round 6's
  original final step (which read the uncommitted `results/tables/*.u16.npy` memmaps and packed the npz) is kept as
  `margin_design.py pack` and refuses to run when a packed npz exists; its log stays `logs/partA1_final.log`.
- DECISION 0-2: "evaluations write per-point incremental checkpoints": `robustness_margin.py sweep` appends one line per
  completed point (its five records, J as exact JSON floats) to `results/partA_sweep.ckpt.jsonl` and `refine_check.py` one
  line per evaluation to `results/partB_refine.ckpt.jsonl` (fsync after each line; not committed); a re-run resumes and
  evaluates only what is missing; the archive / record files are assembled from the checkpoint in the original order and
  format. `--outdir` (redirect outputs) and `--max-new N` (stop after N new points: a deliberate interruption) were added for
  the verification. `equal_send.py` already appended per evaluation and is unchanged.
- DECISION 0-3: Part 0 verification (`part0_verify.py`): `margin_design.py final` once; the sweep interrupted after 20 points
  and resumed; `refine_check.py` interrupted after 5 evaluations and resumed; outputs into `results/runs/part0_verify/` (not
  committed) and compared with the committed round-6 files on every field except wall-clock fields (`wall_s`, `runtime_s`,
  `check_s`), J bitwise.
- DECISION 0-4: report sentences: §0 item 4 of `round6/report_round6_CN.md` is rewritten (2.12 = the actual nominal
  information difference between the two tables, the common baseline cancels; 7.806 vs 7.830 = each table's advantage over its
  own equal-send SoD is similar at its own send level, not "the margin costs 0.024 nat"), with a correction note quoting the
  old wording; a correction record is appended to `round6/decisions_round6.md` (DECISION R-3); the reproduction block of the
  report lists `pack` / `final` and the checkpoints. No number in the report changes.

## Part A (evaluator_refine.py, tables.py, checks.py)

- DECISION A-1: round 4's `evaluator_window.py` is frozen and imported unchanged; the refinement lives in a new module
  `round7/evaluator_refine.py` (a module named `evaluator_window.py` in round7/ would shadow round 4's on `sys.path`). It adds
  only the grid of factor f, the cell lookup, `RefinedTable` and the dispatch; forward, backward, windows88, the assertions and
  the J assembly are round 4's / the accepted evaluator's.
- DECISION A-2: "the accepted evaluator (density_recursion_v1 at n = 24f, Δ* fixed)" on the reuse path is
  `evaluator_fast.evaluate_fast('U', 24 f)`, i.e. `density_recursion_v1.evaluate('B', ...)` with its window path, which the
  speed-up acceptance showed bitwise equal to the full-grid `density_recursion_v1` (report_speedup_CN.md); under mismatch its
  input arrays are (x(θ), H) and its grid is pinned to the nominal grid (round 5's eval_sod; round 6's eval_sod48). Check A2(a)
  compares the dispatched result bit for bit with BOTH evaluate_fast called directly and the full-grid
  density_recursion_v1.evaluate at n = 24, 72, 120.
- DECISION A-3: the grid of factor f is `dr.make_grid` of the NOMINAL output at n = 24 f (z_0 bitwise the coarse z_0; nodes
  z_0 + j h_f). The cell lookup is computed from physical coordinates, u = (r - z_0)/h_24, i = floor(u + 1/2); "no node on a
  boundary" is asserted as |u - i| < 1/2 - 1e-9 (1e-9 cell widths: far above the ~1e-12 floating-point error of u, far below
  the smallest genuine gap 1/10 at f = 5). For odd f the result is cross-checked against the integer rule (j + (f-1)/2) // f.
  On the unshifted σ/48 grid the assertion fires by design (odd nodes on the boundaries); that grid is only used through
  round 4's FineTable (Part B (i)).
- DECISION A-4: a node whose cell lies outside the step's stored range takes the range's boundary node (round 5's runtime rule;
  applied to the cell index). At θ_0 on f = 1 this never happens; under mismatch it is counted (`boundary_steps`, steps where the
  mass support leaves the stored range).
- DECISION A-5: T4's table was not saved (round 6 DECISION A-2: the memmaps are not committed and this clone has none), so one
  DP is re-run at the exact λ of `round6/results/scan_U_W_m020.json` run 2, formed as 0.8 * (5 ḡ) like the scan (checked equal
  to the record), with round 4's `trigger_dp.py` unchanged and 4 numba threads (each node's argmax is independent of the
  thread count). Regressed against the scan record (E[N], S, F, log det, Δm_D, dual value, mass, min m; exact) and round 4's
  U/W scan run at the same λ; packed into `results/table_T4.npz`. Policy IDs: m000 = U_W_r05 (round 4), m005 = U_W_m005_r11,
  m010 = U_W_m010_r11, m020 = U_W_m020_r08, T4 = U_W_m020_r02 (λ/ḡ = 4).
- DECISION A-6: A3's fine grid is z_0 + j h/f, j = 0..200f-1 (G = 200 f, coarse z_0 = -4σ, h = σ/24, h_f = σ/(24 f)), the
  production layout (fine node 0 = coarse node 0); its last (f-1)/2 nodes lie beyond the last coarse cell and take the boundary
  node, exercising the clip. The dense side finds the nearest stored coarse node by brute-force physical distance (unique
  minimiser asserted), takes physical half-widths M^± h, and builds ω from physical offsets measured in fine steps with a
  1e-6 tolerance for the end weights; it does not call `cell_index` or `RefinedTable`. A mismatch count between the lookup and
  the dense nearest node is part of the pass condition.
- DECISION A-7: further regressions in A2 (not required by the spec, kept as evidence): A2(b) also evaluates M = 602 at f = 1
  and compares it with round 6's equal-send archive bit for bit; "f = 1 identity": the new evaluator at f = 1 reproduces round
  6's archive at θ_0 for the four old tables and the SoD bit for bit. The old 24 / 48 regressions are re-run (round 4
  `regression_check`; round 4 C2 / round 6 B-REG numbers in `round4/results/c2.json`).
- DECISION A-8: the 27 points are θ_0 and the 26 selection points read from `round6/results/archive_partA.jsonl` (m020 rows),
  each θ asserted bitwise equal to round 5's `points('U')`; point IDs as round 6 (`cls|dir|sign|eps`).

## Scoring readings for G1–G6 (written before Parts B–D run; the most conservative reading available)

- DECISION S-1 (G1): two sub-claims: (i) |E[N]_48 - E[N]_72| > 1 send at the frozen point (original σ/48); (ii)
  |E[N]_48,shift - E[N]_72| < 0.1 send. The bracket "（支持分界取值解释）" is the reading B2 gives iff (i) and (ii). 对 iff both;
  部分对 if one; 错 if none.
- DECISION S-2 (G2): over the five candidates × 27 points (135 pairs): (i) max|d_1| <= 0.3; (ii) max|d_2| <= 0.05; (iii) the
  median of the signed ratio d_1/d_2 over the pairs with |d_2| > 1e-8 lies in [8, 20]; (iv) "符号一致": sign(d_1) = sign(d_2)
  at every pair (a pair with |d_2| <= 1e-8 counts as not consistent). 对 iff (i)–(iv); 部分对 if one to three; 错 if none.
- DECISION S-3 (G3): (i) on the reference grid m = 0.10 at P1 +0.5 has E[N] > B and the excess E[N] - B rounds to 7 sends
  (6.5 <= excess < 7.5; "约 7 次" read as the nearest integer); (ii) on the reference grid T4 holds all 26 points (E[N] <= B);
  (iii) max over the five candidates and the SoD at Δ*, over the 27 points, of |log det_120 - log det_72| <= 0.01 nat. If the
  C3 gate fails there is no reference grid: (i) and (ii) are then 未定 (not counted as right) and G3 is at best 部分对.
- DECISION S-4 (G4): (i) |second term| <= 0.1 nat for m = 0.20 in the D2 decomposition (未定 if D2 is not applicable or m = 0.20's
  nominal matching fails); (ii) T4's c on the reference grid in [1.6, 1.9] (未定 without a reference grid). 对 iff both;
  部分对 if one; 错 if none (a 未定 sub-claim is not right).
- DECISION S-5 (G5): (i) verdict case 3; (ii) the selection rule (passer of the 26-point budget on the reference grid with the
  smallest c <= 2, ties -> larger E[N](θ_0)) picks T4. 对 iff both; 部分对 if one; 错 if none.
- DECISION S-6 (G6): (i) the counts: T4 27 points at n = 24; 162 forwards at each of f = 3, 5; 841 forwards for the nominal
  curve; 20 × 41 local-matching forwards (memoised overlaps and full rescans reported); (ii) one σ/120 forward "约 10 s":
  median single-forward wall time of the σ/120 evaluations within [5, 20] s; (iii) the two "约 2.3 h" stages: sum of the
  per-forward wall times of the nominal curve and of the local matching each within [1.15, 4.6] h (the estimate is a serial
  one); (iv) total logged wall clock of this round's computations <= 6 h plus, per full rescan, 841 × the median σ/120 forward
  time / 4 workers. A stage that is 不适用 makes its sub-claim 未定. 对 iff (i)–(iv); 部分对 if some; 错 if none.

## Part B (diagnose.py)

- DECISION B-1: (i) "the original σ/48 grid" is round 6 Part B's path itself (`round6/refine_check.eval_table48`: round 4's
  FineTable, nearest coarse node with ties -> smaller index), recomputed and asserted equal to `round6/results/partB_refine.json`;
  (ii) the shifted σ/48 grid has the same node count with nodes z_0 + (j + 1/2) h_48, the new cell lookup (min distance to a
  boundary 1/4 cell), physical half-widths 2 M^± steps, windows centred at the node's own r, support windows windows88 centred
  at x(θ) on it; (iii) σ/72 through the new evaluator. The σ/24 value (round 6 archive) is listed for context only; the reading
  uses (i)–(iii) only.

## Part C (regrade.py)

- DECISION C-1: the M = 480 SoD is evaluated with the five candidates at every point and grid (f = 3, 5 via the reuse path;
  f = 1 from the round-6 archive); its d_1, d_2 are reported separately and are not part of the gate (the gate is "五个候选
  × 27 点").
- DECISION C-2: the f = 1 values of the four old tables and of the SoD are the round-6 archive values (spec: reuse); as a
  regression the new evaluator at f = 1 recomputes all four old tables and the SoD at the 27 points and is compared bit for
  bit (E[N], E[N]_P, S, log det, 81 J elements). T4 at f = 1 is evaluated at the 27 points and its θ_0 value regressed against the scan record.
- DECISION C-3: "|d_2| near rounding" = |d_2| <= 1e-8 send (the forward's relative rounding level is ~1e-13, i.e. ~3e-11 send);
  "阶数未定" is set at a pair if sign(d_1) != sign(d_2) or |d_2| <= 1e-8.
- DECISION C-4: budget pass, ρ_26 and c are computed on each of the three grids; the verdict uses the reference grid only.
  Without a reference grid (gate failed) the σ/120 values are reported as description, labelled 非参考网格, and enter no
  verdict item.
- DECISION C-5: the three main check points of the selected candidate are chosen on the reference grid within the 26
  (largest E[N]_120; lowest Δm_D,120), first in archive order on an exact tie; a point filling two roles is listed once with
  both roles. 精度未定 per C5 is computed for every candidate at every one of the 27 points and reported.

## Part D (match.py)

- DECISION D-1: the nominal curve's M = 480 point is the reuse path (accepted operator at n = 120), equal by construction to
  C2's SoD at θ_0 on σ/120 (checked bitwise); every other M is the general path with the constant window (5M, 5M).
- DECISION D-2: nominal matching and the decomposition D2 are done for all five tables (m = 0 included: its own Q_0; its terms
  are 0 by construction); a table whose nominal match has |residual| > 1 gets decomposition 未定.
- DECISION D-3: D3's local window is [max(120, M_0 - 20), min(960, M_0 + 20)]; its two ends are the endpoints (a clipped end
  counts); the table's E[N] and log det at a P4 point are C2's σ/120 values; every forward is memoised by (point, M) in the
  append-only archive (checkpoint), θ_0's values are the curve's.

## Records (Part A runs)

- DECISION A-9 (record): `checks.py` was run twice. The first run stopped with a KeyError after A-CELL, A2(a), A2(c) and the
  A2(b) dispatch lines had passed: it looked up round 6's equal-send archive with the point id `theta_0`, while round 6 writes
  θ_0 as `theta0|theta_0|+|0.0`. The key was corrected and the whole script re-run; only the second run's log is kept
  (`logs/partA_checks.log`); no number changed between the runs for the lines both produced.
- DECISION A-10 (record): `tables.py` found no T4 checkpoint table and re-ran one DP (24.6 s); every regressed field of the scan
  record and of round 4's run at the same λ is equal bit for bit (`logs/partA0_tables.log`).

## Records (Part C runs)

- DECISION C-6 (record): `regrade.py run` ran once (486 evaluations, 1317.6 s, no assertion stop); `summary` once; `figs` twice:
  the first render of `fig_d1_d2.png` used axes spanning the whole gate band, which squeezed all 135 points into a sliver;
  the axes were set to the data range with the gate stated in text and the figures re-rendered (only the second figs log is
  kept). No number changed.
- DECISION C-7 (record): beyond the spec's 162 + 162 forwards and T4's 27 at n = 24, Part C ran 135 regression forwards at
  f = 1 (four old tables + SoD at the 27 points) and 27 SoD forwards each at f = 3, 5 are among the 162 (6 × 27 counts the SoD).
