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
