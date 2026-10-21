# Decisions — round 2b (overnight mode)

Spec: `reports_CN/plan_round2b_CN.md` v3.2 (frozen, commit 20b14bb). Predictions: `predictions_round2b.md` (commit 4882148).
Where the spec or the task leaves a choice, the most conservative reading was taken and logged here with the alternatives
considered. No decision below adjusts λ to satisfy the budget, changes a frozen tolerance, threshold or the ε_λ refinement
rule, or labels anything certified.

## Part A (2-D / 1-D discrete models)

- DECISION A-1 (Lagrangian constant). The term is `315.892 λ` exactly as written in spec §1.3 and as used in round 2a
  (`round2a/mdp_estimate.py: LAM_BUDGET = 315.892`), not `(B_full − 1) λ = 315.8921168… λ`. Alternative: full-precision
  B − 1. Reason: keeps 3.1/3.2 directly comparable with round 2a's archived 1210.956 and λ*; the difference is
  1.17e-4·λ ≈ 2e-4 in S̃.
- DECISION A-2 (tie rule of the 2-D policy). Ties in the max over the three admissible actions are broken by the round-2a
  rule applied to the resulting velocity: smallest |v| first, then +v before −v. Alternatives: prefer Δj = 0 first; prefer
  deceleration. Affects the stored policy only, never Ṽ_λ (the max value is exact whatever the order); the number of
  states where the chosen action's Q differs from the window max is logged and must be 0.
- DECISION A-3 (policy rows). The memmap holds 4999 rows, k = 2..K (spec: "10^6 × 4999/4"); the row k = K is the
  tie-rule action on Q ≡ 0 (V_{K+1} = 0) and has no effect on any reported number. Alternative: 4998 rows (k = 2..K−1).
- DECISION A-4 (memmap committed as a lossless packed copy). The disk memmap `policy2d.u8` (1 237 597 431 bytes,
  1.1526 GiB) is the working file the forward pass reads; it is listed in `.gitignore`. Committed instead: the lossless
  copy `results/policy2d_xordelta.npz` (row 0 as is, row r XOR row r−1, zlib per row; 5 057 433 bytes) and
  `results/policy2d_meta.json` (λ, shape, byte layout, sha256). Before acceptance the copy is decoded and its sha256
  compared with the memmap's (equal: logs/mdp2d_estimate.log, "A5 packed copy"); `python3 mdp2d_estimate.py unpack`
  rebuilds `policy2d.u8` bit for bit. Alternatives: commit the raw file split into < 100 MB parts (GitHub refuses single
  files > 100 MB; ~1.2 GB of binary history in the shared repo is hard to undo); commit nothing but the sha256 and the
  regeneration command (the session container is ephemeral, so the file would be lost).
- DECISION A-5 (3.2 has two readings). The task compares "1-D DP on the new grid" with round 2a's 1210.956, which is an
  outer-minimised value at ε_λ = 1e-3. Both are reported, each against the 1 % attention line: (a) the new grid at the
  archived round-2a λ* = 1.8335414994039494 (same λ, only the grid changes — isolates the grid effect, since
  1210.956 = 1 + Ṽ_old(λ*) + 315.892 λ*), and (b) the outer-minimised new-grid value from A4 (ε_λ = 1e-2). No pass line
  for either.
- DECISION A-6 (A4 wording "with the archived full-precision λ*"). Read as: each model's S̃_dual is reported together
  with its own minimising λ*, printed with `repr` and archived in `results/*.json`. Alternative reading: evaluate both
  models at round 2a's λ*; that number exists for the 1-D model as 3.2(a) and for both models (unconstrained) as 3.1.
- DECISION A-7 (3.3 for both models). 3.3 is run for the 2-D model (the spec's check) and, with the same tolerance, for
  the 1-D model on the new grid (extra). The open-loop policy in 2-D form is code(i, l_prev) = clip(l_k − l_prev, −1, 1) + 1
  at every state; on the visited column it is exactly l_k (max |l_{k+1} − l_k| = 1 is checked and logged).
- DECISION A-8 (outer loop). Golden section = round 2a's code (same arithmetic), stopping when the bracket width
  ≤ ε_λ; λ* = the evaluated λ with the smallest Ṽ + 315.892 λ. Each Ṽ_λ is cached on disk keyed by repr(λ); a re-run at a
  finer ε replays the cached λ values (the sequence is deterministic), which equals a fresh run at the finer ε.
- DECISION A-9 (T3 definitions). Band: integer test 540 ≤ |i − 1800| ≤ 570 (18 ≤ |δ|/σ ≤ 19, both ends included).
  Low speed: |l| ≤ 15 (|v| ≤ 0.5σ, included). 2-D: v_prev is the state component at step k (k = 2: v_1). 1-D: v_{k−1} is
  the action taken at step k−1, Pr{|v_{k−1}| ≤ 0.5σ} = Σ_i μ_{k−1}(i) 1{|pol_{k−1}(i)| ≤ 15} (k = 2: the initial velocity
  v_1). Time averages are plain means over k = 2..K of the unnormalised probabilities (the forward mass is logged; it
  stays 1 to rounding). The 1-D statistics use the 1-D policy at its own λ* from A4.
- DECISION A-10 (plots). The figure shows Δj(δ, v_prev) ∈ {−1, 0, +1}, which fixes v* = v_prev + Δj·h exactly; a direct
  colour map of v* would look like v_prev itself. Range δ/σ ∈ [−24, 24] (the forward mass lies there); contours of μ_k are
  overlaid.
- DECISION A-11 (concurrency). Part A ran while Part B's four evaluator workers were busy (4 cores). Per-λ times (T6)
  are therefore under load; every timing line carries the load average and the numba thread count.
- DECISION A-12 (extra diagnostic, not a spec check). For the stored λ* policy the forward result is compared with the
  dual value: Ṽ_λ vs (S − 1) − λ(E[N] − 1). It tests the memmap round trip; no pass line is attached to it.

## Part B (search)

- DECISION B-1 (B in F). F uses the frozen full-precision B = 316.8921168493721 (`round2a/config_recheck.json`), the same
  B as the feasibility rule; the task writes it as 316.892. Effect on F ≤ 20 × 1.2e-4. F only steers the search.
- DECISION B-2 (feasibility rule reused, not re-coded). `round2a/recheck7.feasibility` is imported; τ_N = 0.04 from the
  frozen config. Only "按数值判据可行" counts as feasible; the grey zone "边界未定" is not.
- DECISION B-3 (grid starts). "Top 20 by S among the feasible" is ordered by (−S, grid index); top-up by F only if fewer
  than 20 grid points are feasible.
- DECISION B-4 (random starts). `rng = np.random.default_rng(20261022)`; all 20 power vectors first
  (`rng.dirichlet(ones(4), 20)`, uniform on the simplex), then all phases (`rng.uniform(0, 2π, (20, 4))`). Alternative:
  draws interleaved per start.
- DECISION B-5 (old candidates as starts). The seven candidates of `round2a/config_recheck.json` (U-L1657 included,
  although infeasible) enter through x0 = (log(p_1:3/p_4), χ) and the protocol's x → (a, χ) map; the first evaluation of
  each run is therefore that candidate (P5514 reproduces round 2a's S and E[N] bitwise — logged in the summary).
- DECISION B-6 (NM mechanics). scipy minimises −F; `maxfev = 150` is enforced by scipy itself (the run stops at exactly
  150 objective calls — checked in a test and in the per-run log). Duplicated points are evaluated and archived again
  (不去重).
- DECISION B-7 (dense scan). Centre = S_found from the archive of grid + NM evaluations. Phase points: the archived χ_m
  (mod 2π) ± d·π/180, d = 1..6. Power points: p_m + t, the other three multiplied by (1 − p_m − t)/(1 − p_m). If several
  dense points are feasible with S above the centre, S_found is the largest of them; no re-centring.
- DECISION B-8 (window assertion). If the evaluator's window assertion ever fired (AssertionStop), the search would stop
  (STOPPED) rather than archive a fallback value.
- DECISION B-9 (archive layout). Per-evaluation lines in `search_maxS/eval_grid.jsonl`, `search_maxS/nm/eval_<start>.jsonl`,
  `search_maxS/eval_dense.jsonl` (the checkpoints), merged into `search_maxS/archive_all.json`; the dense-scan log is the
  `DENSE` lines of `search_maxS/search_maxS.log`, copied to `search_maxS/dense_scan.log`.

## Part C (reading)

- DECISION C-1 (crossing test). "ρ ± band crosses a branch boundary" is tested on the closed interval
  [ρ − band, ρ + band] against the boundaries 0, 0.10, 0.35; touching a boundary counts as crossing.
- DECISION C-2 (what is refined). Only the 2-D outer minimisation is refined (spec: "re-run the 2-D outer minimisation");
  the band uses the 2-D model's current ε_λ; the 1-D model stays at ε_λ = 1e-2. Both final ε_λ are reported.
