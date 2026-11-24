# verification/sec4: independent cross-check of Section IV

This folder holds an independent re-implementation (numpy/scipy only, no imports from the repository) used
to cross-check the numbers of Sections IV-A (interval triggers) and IV-B (upper bound for SoD-class
triggers). Nothing outside this folder was changed.

## Files

| file | what it is |
|---|---|
| `check_interval.py` | Section IV-A check. Added byte for byte as supplied (sha256 `4f1c236b…cd790`). |
| `check_cert.py` | Section IV-B check. Added byte for byte as supplied (sha256 `5748ffa8…0ab`). |
| `check_cert_min.py` | `check_cert.py` plus an optional fifth argument MIN (Task 3, see below). |
| `results_interval.txt` | stdout of `python check_interval.py` |
| `a_opt.npy` | written by `check_interval.py` itself (`np.save('a_opt.npy', a)`): the optimal a_k of the allocation problem |
| `run_*.txt` | stdout of each `check_cert*.py` run |
| `*.meta` | written by `runlog.py`: command, start/end, wall time, CPU time, peak RSS, exit code |
| `*.err` | stderr of each run; all are empty (0 bytes) |
| `runlog.py` | wrapper used for every run: runs the command with stdout → `.txt` and stderr → `.err`, pins `OMP/OPENBLAS/MKL_NUM_THREADS=1`, writes `.meta` |
| `summarize.py` | parses `run_*.txt` + `.meta` and prints the tables in this README (`python3 summarize.py [--overview] run_*.txt`) |

### How the runs were made

- Environment: Python 3.13.18, numpy 2.5.3, scipy 1.18.1, x86_64, 4 vCPUs, 16 GB.
- Every run was single-threaded (BLAS/OpenMP threads pinned to 1) and started as its own background process from this folder.
- Several runs shared the 4 vCPUs at once, so the wall times below include contention:
  - 19:08–19:20 UTC: four runs (24 full, 12 full, and the two 8 small runs);
  - 19:20–20:29 UTC: four runs (24 full, 12 full, 12 full MIN 0.5, 12 full MIN 0.25);
  - 20:29–21:20 UTC: 24 full alone.
- All runs ended with exit code 0, and no run was stopped early.
- Nothing was tuned. Each script ran with exactly the arguments listed.

## Overview (all `check_cert*.py` runs; round 3 = last tangent-point round)

"smallest bound" is the smallest upper bound over the three rounds; in every run this is the round-3 bound.
"gap" is that bound minus the round-3 best feasible table. "max abs(diff)" is the largest |V* − forward score|
over all 33 backward passes of the run. "all below bound" is True when every "below bound" entry printed by
the script is True (5 entries per run: constant SoD, the three tables, and the one-parameter rule).

| run | grid | edge values | MIN (sigma) | const. SoD E[N] | const. SoD log det J | best table, round 3 | its E[N] | smallest bound | gap | max abs(diff) | one-parameter rule | all below bound | wall time (s) | peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `run_fac8_small.txt` | sigma/8 | 7 + None | 1 | 316.9002 | 197.1995 | 207.5925 | 316.552 | 207.6593 | 0.0668 | 6.8e-13 | 207.4760 (e = 1.5174) | True | 671 | 263.0 |
| `run_fac8_small_min1.txt` | sigma/8 | 7 + None | 1 | 316.9002 | 197.1995 | 207.5925 | 316.552 | 207.6593 | 0.0668 | 6.8e-13 | 207.4760 (e = 1.5174) | True | 679 | 262.8 |
| `run_fac12_full.txt` | sigma/12 | 30 + None | 1 | 316.8952 | 197.2004 | 207.6260 | 316.624 | 207.7023 | 0.0764 | 1.4e-12 | 207.4941 (e = 1.5366) | True | 4110 | 372.6 |
| `run_fac24_full.txt` | sigma/24 | 30 + None | 1 | 316.8922 | 197.2010 | 207.6420 | 316.837 | 207.7114 | 0.0693 | 2.4e-12 | 207.5051 (e = 1.5562) | True | 7958 | 667.4 |
| `run_fac12_full_min0.5.txt` | sigma/12 | 30 + None | 0.5 | 316.8952 | 197.2004 | 207.7096 | 316.480 | 207.8014 | 0.0918 | 1.3e-12 | 207.5662 (e = 1.5569) | True | 4134 | 370.7 |
| `run_fac12_full_min0.25.txt` | sigma/12 | 30 + None | 0.25 | 316.8952 | 197.2004 | 207.7519 | 316.859 | 207.8421 | 0.0901 | 1.4e-12 | 207.5961 (e = 1.5648) | True | 4168 | 370.4 |


## Task 1: interval triggers (`check_interval.py`)

`python check_interval.py > results_interval.txt` took 0.7 s wall time.

| quantity | expected | obtained |
|---|---|---|
| optimum of the allocation problem, log det | 209.0686 | **209.06855** (E[N] = 316.892200, 34 fixed-point iterations) |
| constant half-width a, at the same budget | 1.8578 | **a = 1.8578** |
| log det at that constant half-width | 208.9566 | **208.95658** (0.1120 below the optimum) |

Other lines of the output:
- The bracket for the allocation optimum is [209.06855, 209.06855].
- The optimality condition a_k² γ_k is constant: 1.881594e-02 at both its minimum and maximum.
- a_k ranges over 1.545–2.048, and q_k over 4.06–12.24 %.
- With all samples sent, log det = 219.0085.
- The optimum is 11.8675 above constant SoD (197.2010).
- Section (1): the argmax of c at fixed q matches the symmetric window (−a) to 1e-4, and the numerical and analytic dc/dα agree to the printed digits.
- Section (2): the slope of c_max equals a² to the printed digits.
- Cosmetic: the "table at q=6.3%" line prints `np.float64(0.326)` etc. This is numpy 2's repr, and the values are 0.326, 0.304, 0.269.

## Task 2: certificate check at finer settings (`check_cert.py`, unchanged)

`python check_cert.py 12 full 5 3` and `python check_cert.py 24 full 5 3`. The "full" set has 30 finite edge
values {0.5, 0.75, 1.0, 1.1, …, 3.0, 3.5, 4, 5, 6, 1.25, 1.75, 2.25} plus "no edge", i.e. 31 × 31 = 961
actions per node. Every edge is at least 1σ from the last transmitted value r (hard-coded as FAC nodes).

#### `python3 check_cert.py 12 full 5 3`  →  `run_fac12_full.txt`

- grid sigma/12 (14070 nodes); candidate set: 30 values + None; MIN = 1 sigma (hard-coded: FAC nodes)
- constant SoD: E[N] = 316.8952, log det J = 197.2004

| round | tangent point | best feasible table: log det J | E[N] | upper bound | gap | max abs(diff) in round |
|---|---|---|---|---|---|---|
| 1 | constant SoD | 207.2876 (+10.0872) | 316.358 | 219.7959 (+22.5955) | 12.5083 | 1.4e-12 |
| 2 | table of round 1 | 207.5591 (+10.3587) | 316.889 | 207.8697 (+10.6692) | 0.3105 | 8.0e-13 |
| 3 | table of round 2 | 207.6260 (+10.4256) | 316.624 | 207.7023 (+10.5019) | 0.0764 | 4.9e-13 |

- largest abs(diff) between V* and the forward score, all 33 backward passes: 1.4e-12
- one-parameter rule: `one-parameter rule, e = 1.5366:   207.4941 (+10.2937), E[N] = 316.818   below bound: True`
- 'below bound' entries: 5, all True: True
- wall time: 4110 s (script); wrapper: 4110.5 s wall, 3195.5 s user CPU, peak RSS 372.6 MB, exit code 0, stderr 0 bytes

#### `python3 check_cert.py 24 full 5 3`  →  `run_fac24_full.txt`

- grid sigma/24 (28138 nodes); candidate set: 30 values + None; MIN = 1 sigma (hard-coded: FAC nodes)
- constant SoD: E[N] = 316.8922, log det J = 197.2010

| round | tangent point | best feasible table: log det J | E[N] | upper bound | gap | max abs(diff) in round |
|---|---|---|---|---|---|---|
| 1 | constant SoD | 207.3086 (+10.1076) | 316.841 | 219.8246 (+22.6236) | 12.5160 | 2.4e-12 |
| 2 | table of round 1 | 207.5656 (+10.3646) | 316.651 | 207.8758 (+10.6748) | 0.3102 | 1.9e-12 |
| 3 | table of round 2 | 207.6420 (+10.4411) | 316.837 | 207.7114 (+10.5104) | 0.0693 | 1.1e-12 |

- largest abs(diff) between V* and the forward score, all 33 backward passes: 2.4e-12
- one-parameter rule: `one-parameter rule, e = 1.5562:   207.5051 (+10.3041), E[N] = 316.847   below bound: True`
- 'below bound' entries: 5, all True: True
- wall time: 7958 s (script); wrapper: 7958.3 s wall, 6520.3 s user CPU, peak RSS 667.4 MB, exit code 0, stderr 0 bytes


### Comparison with the repository's own σ/24 numbers

Source: `reports_CN/report_round11b_EN.md` on branch `round11-edge-actions` (Round 11b). It is also quoted in
`prompt_round14_one_parameter_rule_EN.md` and `prompt_round18_accumulated_information_EN.md` on later round branches.

| quantity | repository (Round 11b, σ/24) | this check, `24 full 5 3` (σ/24), round 3 | difference (check − repo) |
|---|---|---|---|
| constant SoD | log det 197.20097668, E[N] = B = 316.8921168 | log det 197.2010, E[N] 316.8922 | equal to the printed digits |
| design table | T′: **207.6467** (σ/24; 207.6445 at σ/120), E[N] 316.843 (σ/24) | best feasible table: **207.6420**, E[N] 316.837 | −0.0047 |
| class bound at budget B | UB_SoD(S ∪ S′; B) = 207.7097 | **207.7114** (budget = this grid's constant-SoD E[N]) | +0.0017 |
| class bound at B + 0.25 (the certificate value **207.715** = 207.7146 quoted in later rounds) | 207.7146 | not computed (the check uses budget B) | (207.7114 − 207.7146 = −0.0032) |
| gap (bound − table) at B | 0.0630 | 0.0693 | +0.0063 |

The two sets of numbers are consistent with each other in this sense:
- The repository's T′ (207.6467) is below the check's bound (207.7114).
- The check's best table (207.6420) is below the repository's bound (207.7097).

Because the classes differ (next list), neither statement is a formal containment argument.

**Differences in setting.** Taken from the Round 11 / 11b reports. I did not compare the two codes line by line.
1. **Action set (class).**
   - The repository's class is S ∪ S′, 884 actions per node:
     - S′ (784 actions): edge values E′ = {0.5, 0.75, 1.0, 1.1, …, 3.0, 3.5, 4, 5, 6, ∞}, minimum distance **1σ** from r.
     - S (100 actions): E = {1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 4, ∞}, minimum distance **2σ** from r (lower edge = min(x̄_k − e⁻σ, r − 2σ), etc.).
   - The check's "full" class has 961 actions per node. Its edge values are E′ ∪ {1.25, 1.75, 2.25}, and *all* actions have minimum distance 1σ.
   - So, in terms of the action definitions, the check's class contains S′, plus 1σ actions with e = 1.25, 1.75, 2.25 that S ∪ S′ lacks.
   - It does **not** contain the repository's 2σ actions (S). When r ∓ 2σ is the active edge, it cannot be written as x̄_k ∓ e σ with a candidate e.
   - So neither class contains the other. Each code also discretises on its own grid (the check's grid starts 14σ below min x).
   - Round 11b reports 88.5 % of T′'s mass on S′-block actions. The rest is on S-block codes, some of which may coincide with an S′ action at that node, because ties go to the S block.
2. **Tangent point.**
   - Repository: a single tangent point, A₁ = J(T₁)⁻¹, where T₁ is Round 11's edge_j1 table (I = 207.3552).
   - Check: starts at constant SoD and re-tangents twice. The round-3 tangent point is J of its own round-2 table (207.5656).
3. **Price search.**
   - Repository: λ₀ = 5γ̄, doubling, ×0.8, then 8 bisections (12 evaluations).
   - Check: λ₀ = 3γ̄, ×0.7 down / ×1.5 up, then bisection until hi/lo < 1.0005, at most 11 evaluations per round.
   - Both report the minimum of the dual over the evaluated λ only.
4. **Budget.**
   - Check: B is the constant-SoD E[N] on its own grid (316.8922 at σ/24, equal to the repository's B to the printed digits).
   - Repository: the quoted certificate 207.715 is at B + 0.25 (Round 11b decision 11b-6). At B itself the repository's bound is 207.7097.
5. **Evaluation.**
   - Repository: T′ is also re-evaluated at σ/120 (I = 207.6445, E[N] = 316.633).
   - Check: evaluates its tables on its own grid only.
   - The check's discretisation follows its own docstring: the continuous c-formula at the edge-node positions, half weight on edge nodes in the send probability and the transitions, and outward rounding of edges. I have not verified that these details match the repository's implementation.

## Task 3: minimum distance MIN between the window edges and r (`check_cert_min.py`)

`check_cert_min.py FAC SET [NBIS] [NRND] [MIN]`:
- MIN is the minimum distance (in σ, default 1.0) of each window edge from the last transmitted value r.
- It must be a positive multiple of 1/FAC, otherwise the script exits with an error.
- M = MIN·FAC grid nodes replaces FAC wherever FAC set that minimum distance:
  - `windows()`: `idx + 1 ∓ M`
  - `backward()`: `iL`, `iU`, and the region boundaries `i_hi = b.max() − M − 1`, `i_lo = a.min() + M`
  - `rule_win()`: `idx + 1 ∓ M`
- Kept unchanged:
  - the constant-SoD half-width `hw = 20*FAC`;
  - the grid step `d = sig/FAC` and the grid margin.
- One line, `MIN = … sigma (… nodes)`, is printed after the grid line.

The full diff against `check_cert.py` is those five code lines, the argument parsing (4 lines), the extra
print, and the docstring.

**Reproduction check (MIN = 1).**
- `check_cert_min.py 8 small 5 3 1` and the unchanged `check_cert.py 8 small 5 3` were run side by side.
- After removing the added `MIN = 1 sigma (8 nodes)` line and the elapsed-time stamps (`(… s)` and `time … s`), the two outputs are identical: `diff` is empty.
- This covers every λ, E[N], log det, V*, forward score, diff, bound, table, and the one-parameter-rule line.
- Both match the reference from your machine: round 3 best table 207.5925, upper bound 207.6593, gap 0.0668.
- The diff column is at most 6.8e-13 in absolute value.

#### `python3 check_cert.py 8 small 5 3`  →  `run_fac8_small.txt`

- grid sigma/8 (9380 nodes); candidate set: 7 values + None; MIN = 1 sigma (hard-coded: FAC nodes)
- constant SoD: E[N] = 316.9002, log det J = 197.1995

| round | tangent point | best feasible table: log det J | E[N] | upper bound | gap | max abs(diff) in round |
|---|---|---|---|---|---|---|
| 1 | constant SoD | 207.2468 (+10.0473) | 316.493 | 219.6807 (+22.4813) | 12.4340 | 6.8e-13 |
| 2 | table of round 1 | 207.5273 (+10.3278) | 316.589 | 207.8429 (+10.6434) | 0.3156 | 6.5e-13 |
| 3 | table of round 2 | 207.5925 (+10.3930) | 316.552 | 207.6593 (+10.4598) | 0.0668 | 4.1e-13 |

- largest abs(diff) between V* and the forward score, all 33 backward passes: 6.8e-13
- one-parameter rule: `one-parameter rule, e = 1.5174:   207.4760 (+10.2765), E[N] = 316.783   below bound: True`
- 'below bound' entries: 5, all True: True
- wall time: 671 s (script); wrapper: 670.9 s wall, 554.7 s user CPU, peak RSS 263.0 MB, exit code 0, stderr 0 bytes

#### `python3 check_cert_min.py 8 small 5 3 1`  →  `run_fac8_small_min1.txt`

- grid sigma/8 (9380 nodes); candidate set: 7 values + None; MIN = 1 sigma (8 nodes)
- constant SoD: E[N] = 316.9002, log det J = 197.1995

| round | tangent point | best feasible table: log det J | E[N] | upper bound | gap | max abs(diff) in round |
|---|---|---|---|---|---|---|
| 1 | constant SoD | 207.2468 (+10.0473) | 316.493 | 219.6807 (+22.4813) | 12.4340 | 6.8e-13 |
| 2 | table of round 1 | 207.5273 (+10.3278) | 316.589 | 207.8429 (+10.6434) | 0.3156 | 6.5e-13 |
| 3 | table of round 2 | 207.5925 (+10.3930) | 316.552 | 207.6593 (+10.4598) | 0.0668 | 4.1e-13 |

- largest abs(diff) between V* and the forward score, all 33 backward passes: 6.8e-13
- one-parameter rule: `one-parameter rule, e = 1.5174:   207.4760 (+10.2765), E[N] = 316.783   below bound: True`
- 'below bound' entries: 5, all True: True
- wall time: 679 s (script); wrapper: 679.7 s wall, 563.5 s user CPU, peak RSS 262.8 MB, exit code 0, stderr 0 bytes

#### `python3 check_cert_min.py 12 full 5 3 0.5`  →  `run_fac12_full_min0.5.txt`

- grid sigma/12 (14070 nodes); candidate set: 30 values + None; MIN = 0.5 sigma (6 nodes)
- constant SoD: E[N] = 316.8952, log det J = 197.2004

| round | tangent point | best feasible table: log det J | E[N] | upper bound | gap | max abs(diff) in round |
|---|---|---|---|---|---|---|
| 1 | constant SoD | 207.3460 (+10.1456) | 316.121 | 220.2423 (+23.0419) | 12.8963 | 1.3e-12 |
| 2 | table of round 1 | 207.6299 (+10.4294) | 316.758 | 207.9869 (+10.7865) | 0.3570 | 3.4e-13 |
| 3 | table of round 2 | 207.7096 (+10.5091) | 316.480 | 207.8014 (+10.6010) | 0.0918 | 3.1e-13 |

- largest abs(diff) between V* and the forward score, all 33 backward passes: 1.3e-12
- one-parameter rule: `one-parameter rule, e = 1.5569:   207.5662 (+10.3658), E[N] = 316.835   below bound: True`
- 'below bound' entries: 5, all True: True
- wall time: 4134 s (script); wrapper: 4134.3 s wall, 3230.5 s user CPU, peak RSS 370.7 MB, exit code 0, stderr 0 bytes

#### `python3 check_cert_min.py 12 full 5 3 0.25`  →  `run_fac12_full_min0.25.txt`

- grid sigma/12 (14070 nodes); candidate set: 30 values + None; MIN = 0.25 sigma (3 nodes)
- constant SoD: E[N] = 316.8952, log det J = 197.2004

| round | tangent point | best feasible table: log det J | E[N] | upper bound | gap | max abs(diff) in round |
|---|---|---|---|---|---|---|
| 1 | constant SoD | 207.3772 (+10.1768) | 316.384 | 220.4238 (+23.2234) | 13.0466 | 1.4e-12 |
| 2 | table of round 1 | 207.6612 (+10.4608) | 316.833 | 208.0348 (+10.8344) | 0.3736 | 3.3e-13 |
| 3 | table of round 2 | 207.7519 (+10.5515) | 316.859 | 207.8421 (+10.6417) | 0.0901 | 2.7e-13 |

- largest abs(diff) between V* and the forward score, all 33 backward passes: 1.4e-12
- one-parameter rule: `one-parameter rule, e = 1.5648:   207.5961 (+10.3957), E[N] = 316.848   below bound: True`
- 'below bound' entries: 5, all True: True
- wall time: 4168 s (script); wrapper: 4168.7 s wall, 3259.9 s user CPU, peak RSS 370.4 MB, exit code 0, stderr 0 bytes


### Effect of MIN at σ/12, full set (round 3)

| MIN (σ) | best table | change vs MIN = 1 | smallest bound | change vs MIN = 1 | gap | one-parameter rule |
|---|---|---|---|---|---|---|
| 1 | 207.6260 | — | 207.7023 | — | 0.0764 | 207.4941 (e = 1.5366) |
| 0.5 | 207.7096 | +0.0836 | 207.8014 | +0.0991 | 0.0918 | 207.5662 (e = 1.5569) |
| 0.25 | 207.7519 | +0.1259 | 207.8421 | +0.1398 | 0.0901 | 207.5961 (e = 1.5648) |

Observations. These are stated as measured, with no tuning; the larger classes were run on σ/12 only.
1. **Class size.** Lowering MIN enlarges the class, and in these runs both the best table and the bound increase.
2. **Tables above the MIN = 1 bound.**
   - The MIN = 0.5 and MIN = 0.25 tables (207.7096, 207.7519) are *above* the MIN = 1 bound on the same grid (207.7023), by 0.0073 and 0.0496.
   - This does not contradict that bound: it holds only for the class with edges at least 1σ from r.
   - It does show that tables with edges closer than 1σ to r reach higher log det than any table of the 1σ class on this grid and budget.
3. **Comparison with the repository's certificate.**
   - The MIN = 0.25 table at σ/12 (207.7519) is also above the repository's σ/24 certificate 207.7146 (by 0.0373); the MIN = 0.5 table is below it (by 0.0050).
   - The grids differ. For MIN = 1, going from σ/12 to σ/24 changed the best table by +0.0160 and the bound by +0.0091.
   - The MIN < 1 classes were not run at σ/24.
4. **Gap.** The gap after three tangent-point rounds is 0.076–0.092 at σ/12, i.e. three rounds do not close it.
5. **One-parameter rule.** Its best e increases slightly as MIN decreases (1.5366 → 1.5569 → 1.5648).

## Notes and things that did not go as expected

- **diff column.**
  - The largest |V* − forward score| is 6.8e-13 (σ/8), 1.4e-12 (σ/12) and 2.4e-12 (σ/24).
  - This is in line with "about 1e-13" from your machine at σ/8. The maximum grows with the number of grid nodes.
- **Wall time.**
  - 12 full took 4110 s (about 68 min) and 24 full took 7958 s (about 2.2 h), with up to four runs sharing 4 vCPUs.
  - 17–22 % of the CPU time was system time (e.g. 24 full: 6520 s user, 1380 s system).
- **Memory.** Peak RSS was 263 MB (σ/8), 371–373 MB (σ/12) and 667 MB (σ/24), all below 1 GB.
- **Feasibility.**
  - "best feasible table" means E[N] ≤ B, with B the constant-SoD E[N] on the same grid. Every reported best table satisfies it.
  - The one-parameter rule is evaluated at e* + 1e-4, so its E[N] is just below B.
