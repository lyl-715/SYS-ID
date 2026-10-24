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
