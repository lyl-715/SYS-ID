# Decisions — round 8b (spec `reports_CN/plan_round8b_CN.md` v2, frozen; amendment to round 8 v3.1)

Each line records a choice the spec (and the task) does not make, or a point where the frozen text cannot be executed as
written, the reading taken and where it acts. Overnight mode: none of these stops the run. Nothing is tuned to a target
using the true error. Wording: 本场景、固定设计规则下的序贯仿真结果.

## P — before the predictions commit

- DECISION P-1: before the predictions commit nothing was computed. The round-8 code, logs, report and archive layout were
  read; the Part 0 text corrections of the round-8 report were written with the numbers the spec gives (medians 2.019e-2 /
  2.509e-7, CP interval [0, 1.22 %]), not re-run; numba 0.68.0 was installed (missing in the container; the version used in
  rounds 6-8; numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2 as in round 8). The Part 0 medians are checked read-only against
  the round-8 archive after this commit (recorded in section A).
- DECISION P-2: the predictions are Q1-Q6 of round 8 v3.1 §4 verbatim (`sed -n '77,84p'`) and Q7 as the round-8b spec words
  it. The engineering-test seed list is part of `predictions_round8b.md` and uses only seeds round 8 already used.

## S — scoring readings for Q1-Q7 (written before any run; the readings of round 8's S-1..S-6, unchanged)

- DECISION S-0: there is one analysis (the primary analysis under the round-8b acceptance rule); each prediction is scored on
  it. 对 iff all sub-claims hold, 部分对 if some, 错 if none.
- DECISION S-1 (Q1): (i) median ε_1 ∈ [0.2, 0.4]; (ii) 90 % quantile ≤ 0.6; (iii) fraction > 0.5 ≤ 15 %; (iv) stage-1
  estimation failure rate ≤ 2 %.
- DECISION S-2 (Q2): (i) |−log det M̂_1 − log det J_1| ≤ 0.5 nat; (ii) whitened Hessian–FIM residual ≤ 0.05.
- DECISION S-3 (Q3): (i) Pr{E[N_2|T,θ_0] > B} ≤ 10 %; (ii) Pr{N_2 > B} ≤ 25 %; (iii) b̂_seq ≤ 2B (point estimates).
- DECISION S-4 (Q4): (i) ΔD over all R ∈ [+2.5, +4] nat; (ii) bootstrap 95 % lower end ≥ +1.5.
- DECISION S-5 (Q5): (i) sequential and (ii) control |−log det M̂ − log det J| ≤ 0.5 nat.
- DECISION S-6 (Q6): (i) design failure rate ≤ 2 %; (ii) every estimation failure rate (stage 1, joint seq., joint control)
  ≤ 2 %; (iii) all 18 per-parameter |bias|/RMSE ≤ 0.2.
- DECISION S-7 (Q7): (i) the median over the 300 replications of the primary work per replication ≤ 110 s, with "primary
  work" defined as in round 8 (worker wall time at one numba thread with 4 workers running: stage 1 and its estimation, the
  control's stage 2, one design, the stage-2 simulation, the post-hoc evaluation, both joint estimates); (ii) Part B + Part C
  wall time as run (Part C includes the bootstrap) ≤ 3 h. This round runs the primary analysis only, so the as-run total is
  the primary-only total.

## A — Part A3 (re-checks under the round-8b acceptance)

- DECISION A-1 (Part 0 numbers, checked read-only after the predictions commit): over the 3000 optimiser calls in
  `round8/results/archive_round8.jsonl` (600 stage 1, 2400 joint) the median terminal ‖∇_ϑ ℓ‖∞ is 2.019e-2 (minimum
  1.658e-4; 0 of 3000 ≤ 1e-8) and the median Newton decrement 2.509e-7, as the spec states. Nothing was changed.
- DECISION A-2 (reuse, no copies): round8b/ imports every round-8 module from ../round8 unchanged (estimator_r8, declared,
  tables_r8, design_r8 with its verbatim mirrors r8k_*, isolation_r8, truth_r8, posthoc_r8, build_kernels). Isolation does not
  require any copy: the design module is round 8's, and its static import graph is checked by round 8's isolation_r8 as it
  stands. So round 8b adds no copy; the 57 mirror blocks of round 8 are still re-extracted and checked byte-equal (R-VERB).
- DECISION A-3 (the acceptance rule): estimator_r8b.accepted(att) is round 8's run_lbfgsb flag accept['supp'] (success,
  all finite, Cholesky of the analytic H_ϑ of the summed NLL, η ≤ 1e-5), i.e. the supplementary analysis S of dcc660b exactly.
  It is re-evaluated from the archived fields (success, finite, η finite and ≤ 1e-5; η stays inf in run_lbfgsb unless H_ϑ is
  finite and its Cholesky factor succeeds) and an assertion requires both to agree on every call. Round 8's 'frozen' flag is
  still computed by run_lbfgsb and archived but plays no role. Seeds are explicit arguments (no module constant patched).
- DECISION A-4 (A3 seeds): the A3 checks use round 8's A3 seeds (listed in the predictions file); the estimator calls inside
  A3 use restart seed 30261028 (r = 0), as round 8's A3 did. No official seed is used.
- DECISION A-5 (A3 pass criteria; round 8's DECISION A-8 tolerances): (a) max_i |θ̂_i − θ̂^LS_i| / s_i ≤ 1e-8 AND the call is
  accepted (round 8's (a) passed vacuously under the frozen rule because the fallback is θ̂^LS; requiring acceptance closes
  that); (b) sqrt(Δθᵀ H_obs Δθ) ≤ 1e-3 standard errors from the archived ML AND accepted; (c) unchanged (relative 1e-6,
  Richardson); check 3 additionally requires the static import graph and the fresh-interpreter import of estimator_r8b to
  reach no θ_0 holder (allowlist {estimator_r8b, estimator_r8, declared, tables_r8, r8k_*}).

## B — pipeline (Part B; written before the official run)

- DECISION B-1 (pipeline entry): round 8's seq_pipeline.one_replication loops over round 8's two acceptance rules and reads
  its seeds from module constants (truth_r8.SEED1/2, estimator_r8.SEED_R1/RJ), so seq_pipeline_r8b.one_replication recomposes
  the same steps for one analysis from the same round-8 functions, with the seed family as a parameter. Engineering test E2
  (round-8 seeds, r ∈ {1, 2, 98, 134}) reproduces round 8's archived supplementary analysis S bit for bit: every archived
  number of the stage-1 calls, design, table codes (sha256), stage-2 records, post-hoc evaluation, H_r and both joint
  estimates, including the two joint fallbacks (r = 134 sequential, r = 98 control), and b_1, J_1
  (`logs/engineering_repro_S.log`, `results/engineering_repro_S.json`). The comparison was checked to detect a one-ulp change.
- DECISION B-2 (noise): e^(1), e^(2) = default_rng(S + r).standard_normal(K), the construction of truth_r8.noise, with the
  official family (S1, S2) = (50261028, 50271028); restart seeds (60261028 + r, 70261028 + r).
- DECISION B-3 (gate, overnight mode): the official run refuses to start unless `results/partA_checks.json` has all
  required checks passed and E2 is all equal. Round 8's B-1, B-2, B-3, B-5, B-6 apply unchanged (first transmission counted
  in N and E[N]; a pre-listed design failure runs the constant-Δ* SoD on the same e^(2); ε_1 with declared.H(); 4 worker
  processes with one numba thread; b_1, J_1 by the reuse path at n = 120). Round 8's B-4 (designs shared between its two
  analyses) has no counterpart. If the official run stops on a correctness failure it is recorded and investigated with
  round-8 seeds only; no official seed is used for debugging.
- DECISION B-4 (archive layout): one JSON line per replication in `results/archive_round8b.jsonl`, round 8's layout with the
  single analysis under the key 'P' (rule 'newton_decrement'), so that round 8's statistics code reads it unchanged; packed
  tables `results/tables_packed/T_r<r>.npz.bz2` (round 8's pack format; the metadata adds the seed family).

## C — statistics (Part C; written before the official run)

- DECISION C-1 (reuse): the statistics are round 8's stats_r8.analyse, called with stats_r8.RES = this round's results
  directory and stats_r8.SEED_BOOT = 50281028 (module parameters set before the call; the code is not changed). Round 8's
  DECISIONs C-1 to C-5 (bootstrap draws, failure rates in the verdict, |bias|/RMSE, Hessian–FIM z and whitening, subset)
  apply unchanged. Engineering test E3 re-computes round 8's analysis S through stats_r8b with seed 20281028 and matches
  round 8's `stats_round8.json` on all 29 quantities compared (ΔD and its interval, subset ΔD, b̂_seq, both proportions,
  failures, information differences, C5, verdict readings and case, M̂; `logs/engineering_stats_E3.log`). The log lines are
  round 8's log_analysis with the analysis label replaced. The bootstrap draws are saved as `results/bootstrap_round8b.npz`.
- DECISION C-2 (Clopper–Pearson): two-sided 95 %, from beta quantiles (scipy.stats.beta: lower = Beta(0.025; x, n − x + 1),
  0 for x = 0; upper = Beta(0.975; x + 1, n − x), 1 for x = n). For 0/300 the interval is [0, 1 − 0.025^(1/300)] =
  [0, 1.2221 %]. A bootstrap interval is called degenerate when its two ends are equal. CP is reported for both over-budget
  proportions (and, extra, for the failure counts); it supplements the frozen verdict rule and never enters it.
- DECISION C-3 (figures, as round 8 with one analysis): ε_1 histogram (with the median and 0.5 marked); E[N_2|T,θ_0] against
  ε_1 with B; box plots of the final errors of both schemes in units of the SoD+SoD CRB standard deviation. Palette: round 8's
  two colours (checked with the dataviz validator: CVD ΔE 24.7, normal-vision ΔE 33.6, contrast ≥ 3:1).
- DECISION C-4 (verdict and scoring): the verdict is the C7 of stats_r8.analyse (spec v3.1 §5, cases 1–4 in order; case 1
  uses this round's A3 result). Q1–Q6 are scored with round 8's verdict_round8.subclaims (readings S-1..S-6), Q7 with S-7.
  The Part B wall time is the sum of the "run finished … s wall" lines of `logs/partB_pipeline.log`.
- DECISION C-5 (engineering tests run before the official run, all on round-8 seeds as listed): E1 = A3 (104.3 s), E2 (4
  replications, 98.2 s wall), E3 (round 8's archive and the E2 output, ~2 s). The reserve E4 was not needed. Their run
  outputs stay under `results/runs/` (not committed); their logs and comparison files are committed.
