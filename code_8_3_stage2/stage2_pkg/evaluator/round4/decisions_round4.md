# Decisions — round 4 (spec `reports_CN/plan_round4_CN.md` v3.1, frozen)

Each line records a choice the spec does not make, the reading taken (the most conservative one available) and where it
acts. Overnight mode: none of these stopped the run.

## Part A (evaluator_window.py)

- DECISION A-1: P5514 = `round2a/config_recheck.json` candidates[0] (`NM-none-P5514`), a and chi at full precision, added to
  the evaluator's in-memory input table only (as `round2a/recheck7.py` does). It is identical to `config.json` input D
  (logged in `logs/partA_checks.log`, line A0).
- DECISION A-2: physical half-widths Delta^+- = M^+- h with h the grid's dz (`dr.make_grid`); alpha = (r_i - M^- dz - x_k)/sigma,
  beta = (r_i + M^+ dz - x_k)/sigma, i.e. the accepted `(z - D - x)/s` with D replaced by the node's own half-width.
- DECISION A-3: in the backward fixed-policy evaluation (and in the DP of Part B) V_{k+1} is taken as 0 at nodes of W_k
  outside W_{k+1}. Those nodes carry zero mass under every table with M^+- <= 960 (support invariant
  R = 40 + 39 + 5 + 4 = 88 sigma; the support-edge assertion checks it on every forward run), so the value of any forward
  run does not depend on this convention; only the stored action at unreachable nodes does.
- DECISION A-4: an assertion failure on the general path (mass, min m, support edge, initial support) stops the run; the
  accepted evaluator's full-grid redo is specific to the constant window and is not carried over.
- DECISION A-5: the small-grid check uses the scenario sigma (s = Delta/20), h = s/24, z_i = -4s + i h, i = 0..199.
- DECISION A-6: small-grid table (c): `default_rng(20261024).integers(6, 61)` drawn for the steps that have a window,
  k = 2..30 (29 x 200: all M^- first, k outer / i inner, then all M^+ the same way); step 1 is the forced send and has no
  window. The random gamma_k in [0.5, 2] for k = 2..30 are drawn next from the same generator (`uniform(0.5, 2.0, 29)`);
  the same gamma is used for all three tables.
- DECISION A-7: the support-edge assertion checks a band of X nodes (the table's largest half-width: 960 at sigma/24,
  1920 at sigma/48) outside the next support window; the accepted evaluator uses M nodes. The band covers every node
  written in the step outside the next window (the window moves by at most 4 sigma = 96 nodes per step).
- DECISION A-8: the small-grid kernel is `dr._kernel` normalised over all 200 nodes (the small grid is the whole support
  window at every step), in both the fast operator and the dense matrices, so the check isolates the transition operator.

## Part B (trigger_dp.py)

- DECISION B-1: objective (ii): J_SoD(u) is Part A's evaluation of the SoD table, which dispatches to the accepted operator
  (bitwise `evaluator_fast`); gamma_k = sigma^-2 ||L^-1 h_k||^2 with J_SoD = L L^T (Cholesky, triangular solve, no inverse
  formed). Check logged: sum_{k>=1} gamma_k w_k(SoD) = tr(J_SoD^-1 J_SoD) = 9.
- DECISION B-2: action code c = i_- * 20 + i_+, (M^-, M^+) = (48 (i_- + 1), 48 (i_+ + 1)) grid units; the SoD action
  (480, 480) is code 189; pad value 65535 beyond each step's support width.
- DECISION B-3: in the DP, c is the formula of `mc83.cfun` evaluated from per-step node lookups u_j = (r_j - x_k)/sigma,
  ndtr(u_j), ndtr(-u_j), phi(u_j) (the scipy/numpy functions cfun uses), with alpha = u_{i-M^-}, beta = u_{i+M^+}; the
  evaluator forms alpha = (r_i - M^- h - x_k)/sigma. The two differ only by rounding of the node position (about 1e-12 in
  alpha). qhat and Phat in the DP are formed by the same code and arrays as in the evaluator (bitwise identical). Logged per
  lambda as a diagnostic: the DP's own value sum_i ghat_1 V_2 against F - lambda N of the forward pass of its stored table.
- DECISION B-4: argmax ties -> the first action in code order (strict ">" over i_- outer, i_+ inner, ascending).
- DECISION B-5: doubling from lambda_0 = 5 gbar runs 5, 10, 20, 40, 80 gbar; a doubled value above the cap 100 gbar is not
  run (the cap is read as a limit on the values run, not as a value to be run).
- DECISION B-6: in the downward scan the first lambda < 1e-3 gbar is run; if it is feasible the scan ends there with the
  frozen message (not reached in any case).
- DECISION B-7: bisection uses the arithmetic midpoint of the last feasible and first infeasible lambda.
- DECISION B-8: feasibility is E[N] <= B with E[N] the Part A forward value (discrete qhat), B = 316.8921168493721; the
  round 2a grey zone B +- tau_N is not used (spec §1.3 item 6).
- DECISION B-9: N in the monotonicity check is N = E[N] - 1 = sum_{k>=2} E[q_k] from the forward pass; all pairs of evaluated
  lambdas are compared after every run; a violation beyond relative 1e-9 would stop that case without a final policy (the
  other cases continue). Not triggered.
- DECISION B-10: each lambda's forward pass is a full Part A evaluation (`evaluate_table`: record and J), so log det and
  Delta m_D are logged per lambda for information; the selection uses only F and feasibility.
- DECISION B-11: checkpointing: after every run the table (uint16 .npy memmap in `results/tables/`, not committed), w_k,
  E[q_k], J (`results/runs/`, not committed) and the scan record `results/scan_<case>.json` (committed) are written; a re-run
  resumes by reusing a run whose lambda matches exactly.
- DECISION B-12: the DP loop over nodes uses numba `prange` (4 threads). Each node is computed independently (no reduction),
  so the tables do not depend on the thread count; the evaluator keeps the accepted flags (no parallel).
- DECISION B-13: the uint16 memmaps of all evaluated tables stay in `results/tables/` (42 MB each, 1.8 GB in all, not
  committed). For each final table the committed copy is `results/table_<case>.npz` (np.savez_compressed: codes, support
  start indices, widths, lambda, run; about 0.3 MB). It is lossless (round trip checked in `logs/final_checks.log`), and
  `np.save('x.npy', np.load('results/table_U_S.npz')['codes'])` rebuilds the memmap.

## Part C (part_c.py)

- DECISION C-1: C1 evaluates each final table with `evaluate_table` (general path; record, J and invariants from the
  accepted `dr.evaluate`). Delta m_D is against the same input's SoD on the same grid. m_E and the nine lambda_i are against
  J_C = uniform-input SoD at sigma/24. Delta m_D against the uniform SoD is logged in addition (`dmD_vs_U`).
- DECISION C-2: sigma/48 reference SoD = `evaluator_fast.evaluate_fast(name, 48)` (accepted operator, R = 68 sigma);
  Delta m_D(48) is relative to it. The sigma/48 budget verdict uses the frozen B = 316.8921168493721 (the uniform SoD's
  E[N] at sigma/24), not the uniform SoD's E[N] at sigma/48 (316.89136, logged). As a diagnostic the general path with the
  constant table (960, 960) at sigma/48 is compared with the accepted operator there (E[N] rel 1.4e-13 / 1.5e-13).
- DECISION C-3: nearest coarse node for fine node j = j // 2. The grids are nested exactly (fine node 2i equals coarse node i
  bitwise, both inputs), so fine node 2i + 1 is an exact-arithmetic tie, which goes to the smaller index. On the
  floating-point positions the two distances of a tie differ by up to 2 ulp (8.9e-16), and a position-based rule would send
  1132 of the 35145 ties (U) to the larger index. The check in `logs/part_c2.log` verifies both facts.
- DECISION C-4: heat maps on (k, (r - x_k)/sigma), every 5th step and every 4th node; nodes with zero forward mass under the
  table are grey (they are never visited, so their stored action is immaterial). E[q_k] and tr(h_k h_k^T A) = h_k^T A h_k are
  drawn on separate axes plus one overlay indexed to each series' mean over k >= 2 (no dual y-axis).
- DECISION C-5 (diagnostic, not required by the spec): Monte Carlo of each final table and of the two SoD tables at the
  runtime lookup rule of spec §1.4 (continuous r; nearest node, ties -> smaller; outside the stored range -> boundary node;
  window centred at the actual r). R = 10000 per row in 20 batches, seeds 20261025 + 1000 row + batch, c = cfun at the
  simulated r (the conditional information, as the stage-2 MC). Reported: E[N], S with sample se; log det of the pooled J
  with a batch-means se (approximate). A first MC run had a bug (the forced first sample's weight w_1 entered as 1/R instead
  of 1); it shifted every log det by about -0.026 and is fixed; E[N] and S were unaffected.
- DECISION C-6 (diagnostic for V4, not required): per final table, the Spearman rank correlation over k = 2..K of the
  mass-weighted mean full width (Delta^- + Delta^+) with gamma_k = sigma^-2 h_k^T A h_k (A = J_SoD^-1 of the same input),
  and, binned by |r - x_k|/sigma, the mass-weighted near-side half-width (the side of x_k as seen from r), the far-side
  half-width and the distance from x_k to the near edge. For context the C3 log also records the mass-weighted share of
  (k, node) with c < 0.01 under the constant threshold (uniform input) and c, q of a single step with the near edge
  2 sigma beyond x_k.
