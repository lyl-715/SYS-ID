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
