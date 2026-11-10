# Round 15 — the one-parameter rule when the sensor's model is an estimate: no correction, margin, feedback

## 0. Why, and rules

A reviewer's second objection: the trigger needs an accurate model of the system that is to be identified, and the current draft assumes the model is known. This round measures what happens to the one-parameter rule of Round 14 when the sensor's model is only as accurate as an estimate from a constant-threshold experiment, and compares two ways of keeping the transmission budget: a fixed margin on the rule's parameter, and a feedback on the number of transmissions sent so far. A small computation made **outside the repository** (12 models, 4000 trajectories) suggested that model error moves the number of transmissions but not the information per transmission, and that the feedback brings the number back to the budget at a cost below 0.5 nat. This round checks that inside the repository with more models.

- Create branch `round15-rule-estimated-model` from the head of `round14-one-parameter-rule`; state that commit in the report. Push only to the new branch. Rounds 4–14 are read-only; new code only in `evaluator/round15/`.
- First commit: this prompt into `reports_CN/`. Second commit: `round15/predictions_round15.md` (§7) before any computation.
- Input `U` only. No DP. Everything except the checks of §5 is Monte Carlo of the continuous rule.
- Assertion failures stop the run. Every choice left open is a `DECISION 15-x` line. Delete any empty folders created under `round4/`.
- This round does **not** run a two-stage experiment (no simulated first stage, no maximum-likelihood estimation). The models are drawn from the distribution such an estimate would have; see §1.

## 1. Models

True parameter `θ*` (the nominal one; `HK @ θ* = XK` to rounding). A model is `θ̃ = θ* + δ`.

**Estimated models.** `δ ~ N(0, c · J_SoD⁻¹)`, with `J_SoD = round4/results/J_U_SoD.npy` (constant SoD at Δ*, `log det` 197.20097668224076), i.e. the Cramér–Rao covariance of an estimate from one constant-SoD experiment (`c = 1`), half of one (`c = 2`), a quarter of one (`c = 4`). With `Lc = np.linalg.cholesky(J_SoD)` (lower), `δ = sqrt(c) · np.linalg.solve(Lc.T, ξ)`, `ξ ~ N(0, I₉)`.
- Main set: `rng = np.random.default_rng(20261112)`; `for c in (1.0, 2.0, 4.0): for d in range(40): ξ = rng.standard_normal(9)` → 120 models.
- Reference set (reproduces the outside computation): `rng = np.random.default_rng(20261111)`; `for c in (1.0, 2.0, 4.0): for d in range(4): ξ = rng.standard_normal(9)` → 12 models.

**Scaled models** (where does the feedback lose authority). Take the 40 directions `δ` of the main set with `c = 1` and rescale each so that its largest prediction error is `ε σ`: `δ_ε = δ · ε σ / max_k |h_kᵀ δ|`, for `ε ∈ {0.5, 1, 2, 4}` → 160 models.

**Gross error.** `θ̃ = (1 ± 0.02) θ*` → 2 models (the prediction is then off by hundreds of σ).

For every model report the prediction error `d_k = h_kᵀ δ / σ`: its root mean square over `k` and its maximum absolute value.

## 2. The rule and its three variants

Prediction `x̄_k = h_kᵀ θ̃`; true output `x_k = h_kᵀ θ*`; measurement `z_k = x_k + σ·noise`; `r` the last transmitted value; array index `p = 1 … K−1` is step `k = p + 1`; the first step always transmits.

```
lo = min( x̄_k − e_k σ , r − σ ),    hi = max( x̄_k + e_k σ , r + σ ),    transmit iff z_k ≤ lo or z_k ≥ hi
information fraction of the step:  c = dr.cfun( (lo − x_k)/σ , (hi − x_k)/σ )        (true output x_k)
```

`W_k` = mean of `c` over the trajectories (`W_1 = 1`), `J = σ⁻² Σ_k W_k h_k h_kᵀ`, `S = Σ W_k`. This is the Fisher information of the record at the true parameter for windows that the receiver knows: it knows `θ̃` and, for the feedback, the number of transmissions received so far.

Nominal parameter `e_0`: the value Round 14 reported for the continuous rule (part 1d, `e_C` rounded up to 0.005 σ; expected 1.580). `B = 316.8921168493721`.

- **V0 — no correction.** `e_k = e_0`.
- **VM(Δ) — margin.** `e_k = e_0 + Δ`, `Δ ∈ {0.05, 0.10, 0.15, 0.20, 0.30}` σ.
- **VF(T, m) — feedback on the count.** With `N_k` the number of transmissions before step `k` (the first one included), target `B_t = B − m`, `q_t = (B_t − 1)/(K − 1)`:
  ```
  e_k = clip( e_0 + ( N_k − (1 + q_t (k − 1)) ) / ( 2 q_t T ) ,  0.5 ,  4.0 )
  ```
  `T ∈ {50, 100, 300, 1000}` steps, `m ∈ {0, 3, 6}` transmissions → 12 settings. `e_k` is computed per trajectory.

## 3. Monte Carlo

Vectorised over trajectories, as `eval11._mc_batch` (noise drawn step by step from a seeded generator, initial `r = x_1 + σ·noise`). One job = one (model, variant): `R = 2000` trajectories, noise seed `20261112` for **every** job (common random numbers across models and variants). The true model (`δ = 0`) is run for every variant with `R = 20000` (10 batches, seeds `20261112 + b`). Up to 4 worker processes.

Per job record: `E[N]` and its standard error, standard deviation of `N` over trajectories, fraction of trajectories with `N > B`, `S`, `log det J`, `ΔI = log det J − 197.20097668224076`; for VF also the mean and standard deviation of `e_k` over (trajectory, step) and the fraction of (trajectory, step) at a clip bound.

Jobs: main set 120 models × 18 variants; reference set 12 models × V0, VF(1000, 0), VF(300, 0), VF(100, 0); scaled and gross-error models × the three variants selected in §4.

## 4. What to report

**4a. Per level `c` and per variant** (main set, 40 models each): over the models — fraction with `E[N] > B`; `E[N] − B`: 5 %, median, 95 %, maximum; `ΔI`: minimum, 5 %, median, 95 %; median standard deviation of `N`; pooled fraction of trajectories with `N > B`. One table per level, variants as rows; the true-model line of each variant alongside.

**4b. Selection (rule fixed now).** A variant *complies* if at most 5 % of the models have `E[N] > B` at `c = 1` **and** at `c = 2`. Among complying margin variants choose the one with the largest median `ΔI` at `c = 2` → `VM*`. Among complying feedback settings choose the one with the largest median `ΔI` at `c = 2` → `VF*`. If no variant of a family complies, say so and take the one with the smallest fraction over budget at `c = 2`.

**4c. Head-to-head.** For V0, `VM*`, `VF*`: the 4a numbers at the three levels, and under the true model. The difference of median `ΔI` between `VF*` and `VM*` at each level.

**4d. Scaled and gross-error models.** For V0, `VM*`, `VF*`: per `ε`, fraction of models with `E[N] > B`, median and maximum of `E[N] − B`, median `ΔI`. For the two gross-error models: `E[N]`, `ΔI`, and for `VF*` the fraction of steps at the upper clip bound.

**4e. One figure** (`fig_round15.png`, one column, font ≥ 8 pt): two panels sharing the x-axis with the three levels `c = 1, 2, 4`; top: `E[N] − B`, bottom: `ΔI`; for each of V0, `VM*`, `VF*` the median as a point and the 5–95 % range over the 40 models as a whisker. Dashed line at 0 in the top panel, at the true-model `ΔI` of V0 in the bottom panel. Caption says what to look at.

## 5. Checks

1. **Outside computation, deterministic part.** For the 12 reference models, evaluate the lattice rule of Round 14 with the *prediction* of the model: `L_i = max(24, i − FLd[p] + n_p)`, `U_i = max(24, CLd[p] − i + n_p)`, `FLd = floor((x̄ − Z0)/DZ)`, `CLd = ceil((x̄ − Z0)/DZ)`, `n_p = 37 if p % 20 < 13 else 38`, through `eval13.forward` with the **true** `XK` (kernel and information). σ/24 values to be matched to 1e-9 relative:

   | c | draw | E[N] | log det J | rms pred. err (σ) | max pred. err (σ) |
   |---|---|---|---|---|---|
   | 1 | 0 | 293.130066163623 | 206.94719294319475 | 0.113856 | 0.243846 |
   | 1 | 1 | 340.7463732234027 | 207.79277045511677 | 0.164476 | 0.374621 |
   | 1 | 2 | 318.4508130092631 | 207.4469157068209 | 0.131573 | 0.288575 |
   | 1 | 3 | 315.63872992664085 | 207.39108535919198 | 0.115342 | 0.249696 |
   | 2 | 0 | 323.5285886160711 | 207.5402691799476 | 0.106571 | 0.228429 |
   | 2 | 1 | 353.8547438394742 | 207.70585912543362 | 0.278381 | 0.641947 |
   | 2 | 2 | 335.7276635272488 | 207.65125283199484 | 0.177686 | 0.398959 |
   | 2 | 3 | 318.6371113922262 | 207.3058071370323 | 0.194471 | 0.473770 |
   | 4 | 0 | 295.0571166894415 | 206.94920759763048 | 0.135837 | 0.299670 |
   | 4 | 1 | 271.76169186770636 | 206.1620347166347 | 0.265468 | 0.589348 |
   | 4 | 2 | 293.2396357394827 | 206.6005178205391 | 0.261610 | 0.542670 |
   | 4 | 3 | 386.8182332082853 | 208.14034340245658 | 0.338782 | 0.706108 |

   If these do not reproduce, stop and report (the construction of the models or of the mismatched rule differs).
2. **Evaluator against simulation under mismatch.** For three of them (`c = 1` draw 1, `c = 2` draw 1, `c = 4` draw 3): σ/120 evaluation of the same lattice rule and a Monte Carlo of that lattice rule at the runtime look-up (Round 14's routine 1c with the half-widths above), `R = 10^4`; z-scores of `E[N]`, `S`, `log det J`; pass rule `|z| ≤ 3`, no clipped look-ups.
3. **True model.** V0 with `R = 20000` agrees with Round 14's continuous-rule result at `e_0` within Monte Carlo error.
4. **Outside computation, Monte Carlo part** (not a gate; report the differences): run the true model once more with `e_0 = 1.5768` for V0 and VF(1000, 0), VF(300, 0), VF(100, 0). With that value the outside run (4000 trajectories, another seed) gave for the true model `E[N] = 316.89`, sd of `N` 18.5, `log det J = 207.505`; with feedback `T = 1000`: 317.07, 5.75, 207.498; `T = 300`: 316.94, 3.14, 207.476; `T = 100`: 317.16, 1.83, 207.421.
5. Every job: `ΔI` finite, `0 ≤ W_k ≤ 1`, `N ≥ 1`.

## 6. Questions the report must answer (fixed before running)

- **Q1** Without correction, what fraction of estimated models exceeds the budget in expectation at each level, and how far does `E[N]` move?
- **Q2** Does the information per transmission survive: is the median `ΔI` of V0 within 0.5 nat of the true-model value at every level?
- **Q3** Is there a complying margin, and what does it cost in median `ΔI` at `c = 2` relative to the true-model V0?
- **Q4** Is there a complying feedback setting, and what does it cost?
- **Q5** At equal compliance, is `VF*` better than `VM*` in median `ΔI`, and by how much, at each level?
- **Q6** What does the feedback do to the spread of `N` over single experiments?
- **Q7** Up to which `ε` do `VM*` and `VF*` keep at most 5 % of the scaled models over budget? What happens at the 2 % gross error?

Answer each with numbers. No causal statements.

## 7. Predictions (commit before running)

```
- Check 1 reproduces to 1e-9; check 2 passes.
- Q1: fraction over budget for V0 between 30 % and 65 % at every level; 5-95 % range of E[N] - B has half-width in
      [20, 50] at c = 1, [30, 70] at c = 2, [40, 100] at c = 4.
- Q2: yes at c = 1 and c = 2.
- Q3: the complying margin is Delta >= 0.10; its median Delta I at c = 2 is at least 0.8 nat below the true-model V0.
- Q4: VF(100, 3) complies; some complying feedback setting has median Delta I at c = 2 within 0.4 nat of the true-model V0;
      true-model cost of VF(T, 0) is <= 0.15 nat for T >= 100.
- Q5: VF* above VM* by at least 0.3 nat at c = 1 and c = 2.
- Q6: median sd of N below 4 for T <= 100 and between 16 and 21 for V0.
- Q7: VF* keeps <= 5 % over budget up to eps = 0.5; at the 2 % gross error E[N] > 2 B for every variant.
Scoring: a line is a hit iff every stated number falls in its range.
```

## 8. Outputs

- `round15/mismatch15.py`, `round15/run_round15.py`, `round15/results/round15.json` (one record per job plus the summaries), `fig_round15.png`, logs.
- `reports_CN/report_round15_CN.md` (+ EN copy): (1) answers to Q1–Q7, one or two lines each; (2) the three level tables; (3) selection and head-to-head; (4) scaled and gross-error models; (5) the true-model lines and check 4; (6) predictions hit/miss; (7) checks with quoted log lines; (8) DECISION list; (9) wall times; (10) not done — in particular: no simulated first stage, no maximum-likelihood estimation, one input, one budget, the feedback law not optimised.
- Runtime guide: one job of 2000 trajectories takes one to two seconds; about 2900 jobs → 15–25 min on four processes; the three σ/120 evaluations of check 2 about 2 min each. Under one hour in total.
