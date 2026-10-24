# Decisions — round 5 (spec `reports_CN/plan_round5_CN.md` v3.2, frozen)

Each line records a choice the spec (and the task) does not make, the reading taken (the most conservative one available)
and where it acts. Overnight mode: none of these stopped the run. Nothing was tuned to a target.

## Part 0 (round-4 maintenance; no DP re-run, no MC)

- DECISION 0-1: `round4/trigger_dp.py` doubling branch: lambda <- min(2 lambda, 100 gbar); the cap value itself is evaluated, and
  only an infeasible cap stops the scan with the frozen message. Sequence 5, 10, 20, 40, 80, 100 gbar (previously the scan
  stopped after 80 gbar). Round-4 results are unaffected: all four round-4 cases were feasible at lambda_0 = 5 gbar.
- DECISION 0-2: the unit test (`round4/test_trigger_dp.py`) replaces `Scan.run` by a synthetic oracle (E[N] a non-increasing
  function of lambda/gbar, F = E[N], run cap 40 kept) and drives the unchanged `Scan.scan`; four cases x gbar in {1, 0.0215873}.
  The step oracle of case (c) switches at lambda/gbar >= 100 - 1e-9 because (100 gbar)/gbar rounds to 99.99999999999999 for
  gbar = 0.0215873 (the first version of the test switched at exactly 100 and failed for that reason, not because of the scan).
  The same test run once against the pre-fix revision (7a837bd) fails cases (a)-(c) for both gbar (logged), so it exercises
  the cap.
- DECISION 0-3: `part_c.py::load_table` reads `round4/results/table_<case>.npz` and asserts that its run number and lambda equal
  the scan's final run and that its start indices and widths equal the design grid's support windows. The MC code reads the
  same npz (changed, not re-run, as instructed; the npz is the lossless copy of the memmap it used).
- DECISION 0-4: verification of 0b in-process (`round5/part0_verify.py`): `part_c.c1` and `part_c.c3` re-run from the npz (the
  memmap folder is absent in this clone), their products compared with the committed files (c1.json field by field except the
  wall-clock eval_s; J_*.npy, c3_table.md, shape_*.json as bytes; the two figures for information), then the committed
  round-4 files written back unchanged, so the round-4 archive and the log lines the round-4 report cites stay as they were.
- DECISION 0-5: information split by branch: the same forward pass twice with `dr.cfun` masked to c^ev, then to c^no (the mass
  recursion does not use c, so m_k and E[N] are bitwise unchanged; checked). The send branch includes the forced first sample
  w_1 = 1, so the two branches add up to S (the reading under which the task's 353.153 + 113.005 = 466.158).
- DECISION 0-6: mean send probability = (E[N] - 1)/(K - 1) (forced first sample excluded); full-trajectory ratio
  (S - 1)/(E[N] - 1); the single-step c/q is computed for alpha = -2, beta = 30 only and labelled as an illustration.

## Part A (robustness.py)

- DECISION A-1: theta_0 = [G(0), Re G(j w_1), Im G(j w_1), ..., Re G(j w_4), Im G(j w_4)] with G(s) = 2/(1+s)^2 (mc83.py), the
  layout of the regressor h_k = [a0, a_m cos psi_m, -a_m sin psi_m]. The nominal output stays the round-4 array xbar
  (dr.input_arrays); x(theta) = xbar + H (theta - theta_0) with theta - theta_0 taken from the archived theta vector. H theta_0
  is computed only as a logged diagnostic of the layout (max |H theta_0 - xbar| / sigma), never used.
- DECISION A-2: support windows under mismatch: the round-4 formula (windows88 for the tables, R = 88 sigma;
  evaluator_fast.windows for the SoD, R = Delta + 48 sigma) centred at x_k(theta) on the frozen nominal grid. They are a
  computational device that must hold the mass, whose support moves with x_k(theta); the table's stored ranges (start, width)
  stay frozen and a node outside its step's stored range takes the boundary node's action (the frozen lookup rule). Each table
  evaluation logs how many steps actually used a boundary lookup with non-zero mass. The grid must contain every window plus
  the largest half-width (asserted).
- DECISION A-3: the constant-Delta SoD control at theta is evaluator_fast.evaluate_fast with dr.input_arrays returning
  (x(theta), H) and dr.make_grid pinned to the nominal xbar for that call (so the SoD and the tables share the frozen grid);
  the table evaluations use dr.evaluate with dr.input_arrays and dr.run_ab replaced for the call (as round 4 replaced
  run_ab). All replacements are restored after each call; workers run one evaluation at a time.
- DECISION A-4: P1 uses the per-input normalisation max_k |h_k^T e_1| = a0 = 1, i.e. theta_1 += sign eps sigma exactly as the
  spec writes it. P2: eta = sign eps sigma / max_k |h_k^T (0, theta_0[2..9])| (the AC part of the nominal output). P3 includes
  theta_1 (i = 1..9, 18 points; its +-0.5 points coincide with P1 at eps = 0.5 and are evaluated again as P3 points). P4: one
  sign (theta_0 + eps sigma d / max_k |h_k^T d|, as the spec writes it); d_j = row j of
  default_rng(20261025).standard_normal((10, 9)) normalised to unit Euclidean length.
- DECISION A-5: an assertion of the accepted evaluator (mass, min m, support edge, initial support) stops that evaluation (no
  fallback, as round-4 DECISION A-4); it is archived with status 'assert_stop' and the sweep goes on with the other
  evaluations. Counted in the report.
- DECISION A-6: the theta_0 regression evaluates all four tables (P5514/S included) and both SoDs through the mismatch code path
  with theta = theta_0, and compares: tables with round4/results/c1.json (E[N], E[N]_P, S, log det, Delta m_D: exact equality)
  and with round4/results/J_<case>.npy (81 elements, exact), and w_k, E[q_k] with the round-4 path (evaluator_window
  .evaluate_table with CodeTable) run in the same process; SoDs with evaluator_fast.evaluate_fast run unchanged (E[N],
  E[N]_P, S, log det, 81 J elements, 5000 w_k: exact).
- DECISION A-7: the sweep evaluates the three tables of spec §1.2 (U/S, U/W, P5514/W) and the two SoDs at every point
  (400 evaluations); P5514/S only at theta_0. Delta m_D of a table = log det J_table(theta) - log det J_SoD(theta) of the same
  input at the same theta; retention = Delta m_D(theta) / Delta m_D(theta_0) of the same table. The budget verdict is
  E[N] <= B with the discrete evaluator's E[N], labelled 离散方案下; for the SoD rows it is recorded too.

## Scoring readings (written before the Part A sweep and the Part B runs; the most conservative reading available)

- DECISION S-1 (g_med, eps_50): g_med(eps) = median over the ten P4 directions of the U/W table's Delta m_D at the sampled
  eps in {0.2, 0.5, 1, 2}; the nominal value is Delta m_D(theta_0) = g_med(0). "Below half" = g_med(eps) < g_med(0)/2.
  eps_50 = the first sampled eps below half, reported with its two neighbouring sampled points (the previous one may be 0).
  A crossing is a change of the predicate between consecutive sampled points (0 included). 非单调 is flagged when g_med is not
  non-increasing over 0, 0.2, 0.5, 1, 2 or crosses more than once; then every crossing is reported. The same quantities are
  logged for U/S and P5514/W as supplementary lines.
- DECISION S-2 (eps_B): per table, for each P1 sign the sampled eps in {0.1, 0.2, 0.5, 1, 2, 4} at which E[N] first exceeds B
  (eps = 0 is within budget for all tables). Worst sign = the sign with the smaller first-crossing eps; equal or both without
  crossing -> the sign with the larger E[N] at the largest sampled eps at which they differ (both reported anyway). Same
  neighbour / crossing / 非单调 rules as S-1 (non-monotone = E[N] not monotone in eps along that sign). Headline: U/W; U/S and
  P5514/W reported with it.
- DECISION S-3 (R1): 对 iff g_med(0.5) < g_med(0)/2 and g_med(2) <= 0; one of the two -> 部分对; none -> 错.
- DECISION S-4 (R2): the spec does not name the table, so all three swept tables are scored; for a table R2 holds iff at some
  sampled eps <= 1 one P1 sign has E[N] > B and the opposite sign at the same eps has E[N] < 200. 对 iff it holds for all three
  tables; 部分对 if for some; 错 if for none.
- DECISION S-5 (R3): per table, holds iff at P4 eps = 0.2 at least 8 of the 10 directions have Delta m_D >= 0.7 Delta m_D(theta_0)
  and E[N] <= B (both on the same direction). U/W decides the reading of spec §4 (it is the table of the two-stage scheme B);
  the verdict line also gives U/S and P5514/W, and R3 is 对 only if it holds for all three, 部分对 if for U/W but not all, 错 if
  not for U/W.
- DECISION S-6 (R4): four sub-claims: (i) log det B - log det A >= 3 at theta_0; (ii) log det D - log det B <= 1 at theta_0;
  (iii) median over the ten P4 eps = 0.2 directions of [log det(J_SoD + J_UW) - log det(2 J_SoD)] >= 3; (iv) at least 8 of those
  directions with E[N_SoD] + E[N_UW] <= 2B. 对 iff all four; 部分对 if some; 错 if none. The B - A' = 4.852 check is not scored.
- DECISION S-7 (R5): retention = Delta m_D(theta)/Delta m_D(theta_0). Compared at the same eps level, P5514/W against U/W (the
  only P5514 table swept; both weighted): P1 and P2 at each signed eps (12 + 10 points), P3 by the median over its 18 points,
  P4 by the median over the ten directions at each eps (4). 27 comparisons; 对 iff P5514/W's retention is lower in all, 部分对
  if in some, 错 if in none. The 80 pointwise comparisons are logged as a supplement.
- DECISION S-8 (R6): wall time of the Part A sweep (400 forwards) + scheme A's 241 forwards + scheme D's lambda scan, as
  logged; 对 iff <= 1.5 h. The theta_0 checks and figures are logged separately and not counted.

## Part B (two_stage.py)

- DECISION B-1: scheme A evaluates every integer M = 240..480 with round-4 `evaluate_table('U', ConstTable(M, M))` at theta_0:
  the general window path (R = 88 sigma) for M < 480 and, by round 4's frozen dispatch rule, the accepted operator
  (evaluator_fast) at M = 480. M_A = argmax{E[N_M] : E[N_M] <= 2B}, ties -> larger M. If the slack 2B - E[N_M_A] exceeds 5
  sends it is reported as it is and the range is not widened (spec §1.3).
- DECISION B-2: A' and B at theta_0 use the round-4 J's (round4/results/J_U_SoD.npy, J_U_W.npy; bit for bit equal to the Part A
  theta_0 check) and E[N]'s (c1.json). log det of a sum of information matrices is taken as given (two independent
  experiments at the same theta); the sends of the two stages are added. For the single-stage schemes A and D the
  "each stage <= B" constraint does not apply (logged as n/a); their total is judged against 2B.
- DECISION B-3: scheme D reuses `round4/trigger_dp.py` unchanged (Design, gamma('W') from the same J_SoD, Scan with the Part 0a
  doubling rule). Its module-level budget and output paths are pointed at round 5 for this run only (td.B = 2B; tables, runs and
  the scan record under round5/results/; case_paths -> scan_U_W_2B), so no round-4 file is written. The DP's every lambda is
  in the scan record and the log; the uint16 memmaps stay uncommitted (as round 4), the final table is packed losslessly into
  results/table_U_W_2B.npz.
- DECISION B-4: the P4 rows of Part B (eps in {0.2, 0.5}, ten directions) take the U/SoD and U/W evaluations of the Part A
  archive at the same theta (same J's, from results/J_partA.npz); nothing is re-evaluated. A' total = 2 E[N_SoD(theta)],
  B total = E[N_SoD(theta)] + E[N_UW(theta)], each against 2B; "each stage <= B" logged separately.

## Part C (headline.py, report)

- DECISION C-1: the 非单调 flag of S-1 fires for all three g_med curves because they are not non-increasing: they rise
  monotonically with eps (U/W 7.830, 7.855, 7.885, 7.942, 8.179) and never cross half the nominal value. The flag is reported
  as the pre-written reading gives it, together with this explanation; in the ordinary sense the curves are monotone
  (increasing) and have no crossing.
- DECISION C-2: B is by definition the uniform input's constant-Delta SoD E[N] at theta_0, so U/SoD sits exactly on B; under P1
  its E[N] moves by at most 1.2e-11 (shift invariance of the constant threshold), and its budget verdicts flip at that
  rounding level (8/12 "> B"). They are recorded as the discrete evaluator gives them but are not a budget signal; no
  prediction is scored on the SoD rows. The same definition puts A' exactly on 2B at theta_0 (slack 0.0), so at the P4 points
  A' exceeds 2B in 6/10 directions by at most 0.37 sends (real discrete changes of E[N_SoD(theta)], recorded as given).
- DECISION C-3: under mismatch the table and the paired SoD no longer send the same expected number of samples (the table's
  E[N] leaves B, the SoD's stays within 316.9 +- 1.1); Delta m_D is reported as the spec defines it (same theta, different
  send counts) and the report says so. No equal-send-count comparison under mismatch is computed (not in the spec). For scale
  the report quotes scheme A's curve at theta_0 (log det of the constant-Delta SoD against its own E[N]); that is context,
  not a correction.
- DECISION C-4: headline.py gained one log line (C-SOD-P1: the SoD controls' largest E[N] deviation over the P1 points)
  after its first run, to make C-2 traceable; nothing else changed and no scored line moved.
- DECISION C-5 (record): the first run of part0_verify.py built the non-send mask from the already replaced dr.cfun, so the
  non-send branch came out as 0 (the send branch and every 0b comparison were already right). The masking function now
  receives the original cfun; the script was re-run and only the second run's log is committed. Reported in the report's
  run-time paragraph.
