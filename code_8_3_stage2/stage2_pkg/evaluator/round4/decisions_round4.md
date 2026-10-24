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
