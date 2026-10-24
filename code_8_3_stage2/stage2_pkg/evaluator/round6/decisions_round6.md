# Decisions — round 6 (spec `reports_CN/plan_round6_CN.md` v3.2, frozen)

Each line records a choice the spec (and the task) does not make, the reading taken (the most conservative one available)
and where it acts. Overnight mode: none of these stopped the run. Nothing was tuned to a target. Wording: 固定设计、离散方案下;
nothing here is a bound.

## Part 0 (round-5 fixes; no redesign, no MC)

- DECISION 0-1: 0a re-emits the three g_med statements of `round5/logs/partC_headline.log` (lines 7, 12, 17) as
  "单调上升，未观察到减半跨越（测试至 ε=2）" (the task's phrase plus the tested range, which the old line also carried). The
  monotonicity test is two-sided over the sampled points 0, 0.2, 0.5, 1, 2 (non-strict: equal neighbours do not break
  monotonicity). The three corresponding statements of `round5/report_round5_CN.md` are corrected in place and marked as a
  round-6 correction; the correction note is appended to `round5/decisions_round5.md` (S-1 and C-1 point to it).
- DECISION 0-2: 0b's "redesign from an empty checkpoint" (`two_stage.py schemeD-redesign`) refuses to start, before any DP,
  when the scan record `results/scan_U_W_2B.json` or any `results/{tables,runs}/U_W_2B_r*` file exists. It was run once only
  to exercise that guard (it stopped on the committed scan record; `round5/logs/partB_schemeD-redesign.log`); no redesign.
- DECISION 0-3: 0b's loader (`two_stage.py schemeD-load`) checks with `git ls-files` that each file it reads is committed
  (scan record, packed table, schemeD.json, J_schemeD.npy) and asserts the npz's run number and lambda against the scan
  record and schemeD.json and its support ranges against the design grid. It was verified in this clone, where
  `round5/results/tables/` and `round5/results/runs/` do not exist, so no uncommitted table could have been read.

## Part A (margin_design.py, robustness_margin.py)

- DECISION A-1: `round4/trigger_dp.py` is reused unchanged; for the scan of margin m its module-level budget is set to
  B_m = (1 - m) B (float product, logged with repr) and its tables/runs/scan record are pointed at `round6/results/`
  (as round-5 scheme D did for 2B). Everything that reads the budget inside trigger_dp then uses B_m: feasibility
  E[N] <= B_m, the slack B_m - E[N] and the dual value g_1 + sum ghat_1 V_2 + lambda (B_m - 1) logged per run. The three
  scans share one `Design('U')` (same grid, same cache, same gamma) and run one after another with 4 numba threads; each
  has its own run record, run cap 40 and checkpoint; no DP result is shared between margins even where a lambda value
  coincides (each scan is the round-4 rule applied on its own).
- DECISION A-2: the final tables are packed losslessly into `round6/results/table_U_W_m{005,010,020}.npz` (codes, support
  start indices, widths, lambda, run, m, B_m; round trip checked). The uint16 memmaps of every evaluated lambda stay in
  `round6/results/tables/` and are not committed (42 MB each, as rounds 4 and 5); every later step reads the npz only.
- DECISION A-3: "with round 4's U/weighted table (m = 0)": the m = 0 row is round 4's final U/W table
  (`round4/results/table_U_W.npz`, run 5, lambda 3.4 gbar), read, not re-designed; its scan record and forward-backward
  check are quoted from `round4/results/scan_U_W.json` and `final_checks.json`.
- DECISION A-4: the 80 points and the evaluation are round 5's code itself (`round5/robustness.py` imported, unchanged):
  `points('U')`, `x_of`, `LookupTable`, `eval_table`, `eval_sod`, `evaluate_point`, with the four tables passed in its table
  dictionary under the names m000, m005, m010, m020. The paired constant-Delta SoD is evaluated again at every point
  (`eval_sod`) rather than copied from round 5; as a regression the m = 0 table and the SoD are compared with round 5's
  archive at all 80 points (E[N], E[N]_P, S, log det, 81 J elements; exact equality). theta_0 is evaluated through the same
  path (`x_of(theta_0)`) for all four tables and the SoD; check 1 = the m = 0 table and the SoD at theta_0 equal round 5
  (`round5/results/partA_check.json`: E[N], S, log det, Delta m_D) and round 4 (`round4/results/J_U_W.npy`, 81 elements) bit
  for bit.
- DECISION A-5: Delta m_D(theta) = log det J_table(theta) - log det J_SoD,480(theta) (same theta, the uniform input's SoD at
  Delta* = 480 h); retention = Delta m_D(theta) / Delta m_D(theta_0) of the same table; rho_26 = min of the retention over the
  26-point selection set. The budget verdict is E[N] <= B (the TRUE B, not B_m) with the discrete evaluator's E[N], labelled
  离散方案下.
- DECISION A-6: the selection set is the 20 P4 points with eps_target in {0.2, 0.5} (ten directions each) and the 6 P1 points
  with eps_target in {0.1, 0.2, 0.5}, signs + and -. Candidates for m* are the three margin tables (spec §1.3: "三档都不满足"
  refers to the three margins); the m = 0 table's 26-point verdict is reported as well and is not a candidate. m* = the
  smallest m in {0.05, 0.10, 0.20} whose 26 points all have E[N] <= B.
- DECISION A-7: per-table report rows (spec §1.3, task A5): "min slack min(B - E[N])" is given over the 26 points (the
  selection set) and, separately, over the 80 points (= B - max E[N] over the 80); g_med(eps) = median over the ten P4
  directions of Delta m_D at eps in {0.2, 0.5, 1, 2}, reported with g_med(0) = Delta m_D(theta_0) and the round-6 Part 0a
  monotonicity rule (two-sided, non-strict).

## Scoring readings (written before the Part A runs; the most conservative reading available)

- DECISION S-1 (M1): 对 iff for each of the three margin tables B_m - E[N](theta_0) <= 0.5 (and E[N](theta_0) <= B_m);
  部分对 if for one or two; 错 if for none. Not a scientific finding (spec).
- DECISION S-2 (M2): four sub-claims, all needed for 对: (i) m* = 0.10; (ii) the m = 0.05 table has at least 2 of the 26 points
  with E[N] > B, and every one of its 26-point crossings lies in P4 eps = 0.5 or P1 +0.5 (the locations the prediction names);
  (iii) the m = 0.10 table holds all 26; (iv) the m = 0.20 table holds all 26, the ten P4 eps = 1 directions and P1 +-1.
  Some but not all -> 部分对; none -> 错.
- DECISION S-3 (M3): per margin, Delta m_D(m; theta_0) >= Delta m_D(0; theta_0) - 6m with the exact round-4 value
  Delta m_D(0; theta_0) = 7.830436416895992 (the printed thresholds 7.5304, 7.2304, 6.6304 are its rounding; a value between
  the exact and the rounded threshold would be reported as such). 对 iff all three; 部分对 if one or two; 错 if none.
- DECISION S-4 (M4): (i) rho_26 >= 0.8 for each of the four tables; (ii) for each of the four tables g_med is non-decreasing
  over eps = 0.2, 0.5, 1, 2 (the points the prediction lists; with eps = 0 included it is reported as a supplement). 对 iff
  (i) and (ii) for all four; 部分对 if some of the eight table-claims hold; 错 if none.
- DECISION S-5 (M5): for the selected table's three main grid-check points: |E[N]_48 - E[N]_24| <= 0.5 and the budget
  verdict (E[N] <= B) the same on both grids. 对 iff all three points satisfy both; 部分对 if some; 错 if none.
- DECISION S-6 (M6): (i) all 21 points matched (residual <= 1 send by the C2 rule); (ii) median of Delta m_D^eq over the
  20 P4 points >= 6.0; (iii) median of Delta m_D - Delta m_D^eq over the 20 P4 points <= 1.5 nat. (ii) and (iii) are
  scored only if all 20 P4 points are matched; otherwise they are 未定 and M6 is at best 部分对. 对 iff (i)-(iii) hold.
- DECISION S-7 (M7): total = the sum of the logged wall times of every computation of this round (Part 0 verification,
  three lambda scans, the final checks, the Part A evaluations, Part B, Part C, the figure scripts); 对 iff <= 2.5 h. The
  elapsed time from the predictions commit to the report commit is reported alongside.
- DECISION A-8 (record): the three scans were run in one process (`margin_design.py scan`, 591 s); each started at lambda_0 =
  5 gbar, which every margin found feasible (E[N] = 213.23), so no doubling ran. m = 0.05 and 0.10 bisected between 4 and
  3.2 gbar, m = 0.20 between 5 and 4 gbar (its 4 gbar run, E[N] = 262.80, is above B_0.20 = 253.51); 11, 11 and 10 DP runs,
  no run cap, no monotonicity violation. The DP runs at lambda values shared with round 4 (5, 4, 3.2, 3.6, 3.4 gbar)
  reproduce round 4's E[N] bit for bit (e.g. 3.4 gbar: 316.6035659260541, the round-4 final table), as a deterministic DP must.

## Part B (refine_check.py)

- DECISION B-1: round 5's mismatch conventions carried to the sigma/48 grid: the frozen fine grid is `dr.make_grid` of the
  NOMINAL output at n = 48 (nested exactly in the frozen sigma/24 grid; checked), support windows `windows88` centred at
  x(theta) on it, the table through `evaluator_window.FineTable` (unchanged), record and J from the accepted `dr.evaluate`
  with its input arrays and run_ab replaced for the call (as `robustness.eval_table`). The sigma/48 SoD at theta is
  `evaluator_fast.evaluate_fast('U', 48)` with input arrays (x(theta), H) and its grid pinned to the nominal fine grid (as
  `robustness.eval_sod`; round 4 C2's reference SoD at theta_0). Delta m_D on the fine grid is against that SoD at the same
  theta (round 4 DECISION C-2); budget verdicts on both grids against the frozen B.
- DECISION B-2: the main check points are chosen on the selected table's sigma/24 values from the Part A archive (largest
  E[N] and lowest Delta m_D within the 26-point set; the first in archive order on an exact tie). A point that fills two roles
  is evaluated once and carries both roles; the supplementary points (largest E[N], lowest Delta m_D among all 80) get the
  same three tests, labelled 补充、不参与判定.
- DECISION B-3: path regression: the m = 0 table and the SoD at theta_0 through this mismatch sigma/48 path are compared with
  round 4's C2 numbers (`round4/results/c2.json`: E[N]_48, S_48, log det_48; exact).

## Part C (equal_send.py)

- DECISION C-1: the constant-Delta SoD at half-width M: M = 480 -> `robustness.eval_sod` (evaluator_fast, the accepted
  operator, i.e. the SoD of Part A's Delta m_D), M != 480 -> `robustness.eval_table` with the constant window (M, M) (round-4
  general path, R = 88 sigma); round 4's dispatch rule, as round 5's scheme A. The constant window is passed as a subclass of
  `ConstTable` carrying start = 0 and width = G only for eval_table's boundary-lookup diagnostic (always 0 for it). So
  Delta m_D - Delta m_D^eq = 0 exactly when M_eq = 480.
- DECISION C-2: the window endpoints are the two ends of the evaluated window [max(120, M_0 - 20), min(960, M_0 + 20)]
  (a clipped window's end at 120 or 960 counts as an endpoint). After a full rescan the minimiser over [120, 960] (ties ->
  larger M) is accepted iff its residual is <= 1 send (the spec sets no endpoint condition after the rescan); an unmatched
  point is reported with its closest M and residual.
- DECISION C-3: every forward is memoised by (point, M): theta_0's local window consists of its curve values, a full
  rescan re-uses the point's local-window values; the archive `results/archive_partC.jsonl` holds one line per distinct
  evaluation and is appended as each completes (checkpoint; a re-run resumes from it).
- DECISION C-4: E[N]_table(theta) and log det J_table(theta) are the Part A archive values of the selected table at the same
  theta (same evaluation); the 20 P4 points are those of the selection set (eps 0.2 and 0.5). theta_0 is matched and
  reported as one of the 21 points; verdict items (c) and (d) use the 20 P4 points only (spec §6).
- DECISION C-5: regressions: the theta_0 curve at M = 240..480 against round 5's scheme A rows (E[N], S, log det; exact), and
  M = 480 at theta_0 against Part A's SoD at theta_0.

## Report (verdict_round6.py, report_round6_CN.md)

- DECISION R-1 (record): `robustness_margin.py figs` was run twice: the first render had Chinese text in one axis title, which
  the figure font cannot draw (glyph warnings); the title was changed to English and the figures re-rendered. Only the
  second run's log is kept (`logs/partA_figs.log`); no number changed. The Part C figure title was made English before its
  first render.
- DECISION R-2: M6 sub-claim (iii) is scored by its literal threshold (median of Delta m_D - Delta m_D^eq <= 1.5): -2.039
  satisfies it, so M6 is 对; the report states that the sign is the opposite of the direction the prediction implies
  (the advantage grows after matching, it does not shrink).
- DECISION R-3: the report's §0 item 4 places Delta m_D^eq(theta_0) of the m = 0.20 table (7.806, equal-send baseline) next to
  the m = 0 table's nominal Delta m_D (7.830, whose SoD sends almost the same number) as a note on the dependence of the
  "nominal cost" on the Delta* baseline; it is not used to override the frozen §6 rule, and the verdict stands as case 1.
