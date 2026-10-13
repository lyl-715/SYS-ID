# §8.5 stage 4: predictions (written before any stage-4 run; 2026-10-13)

These predictions were committed before the first stage-4 script ran. They use only results already in `stage2/` and `stage3/`, plus the crude models stated below. No stage-4 noise was drawn and no stage-4 candidate was evaluated. The only stage-4 quantity computed beforehand is non-random: the multiplicity-corrected critical value t_{19, 1−0.05/(2M)}, which is 4.55 for n = 20 (M = 228) and 4.77 for n = 26 (M = 375).

Notation follows stages 2–3. Δm_D = Δm_D^(C) in nat. "Best" means best evaluated qualified candidate at R = 2000 (seed 20261012). "Point > 0.02" means the paired difference new − previous has its point estimate above 0.02 nat. "Lower > 0.02" means the lower end of its exploratory 95% interval (10-batch t, 9 df; `design.md` §1.6) is above 0.02.

## 1. Script 1: change of the front after the added search

**What is added.** Per floor: 12 Nelder–Mead runs × 300 evaluations at R = 500, i.e. 3600 search evaluations. Stages 2–3 together spent 450 + 1000 (none), 450 + 800 (0.3), 450 + 1000 (0.5) and 450 + 800 (0.7) for these floors. So the added budget is 2.5–3 times the previous NM budget per floor.

**Basis.**
- In stage 3, restarts from front points changed Δm_D by −0.010 to +0.008 at R = 2000. The only large move (+0.237 at λ0 = 0.7) came from a new start near the floor (P4055, m_E 0.668).
- Half of the new starts (6 per floor) come from the qualified R = 200 pool. Its points have Δm_D^(C) mostly below +0.26 and m_E below 0.59. In 300 evaluations in 7 dimensions I expect these runs to reach the budget boundary but not the current front.
- The front has a gap in m_E between 0.146 (P4312) and 0.421 (NM-0.3-P4312). The λ0 = 0.3 runs started from the middle stratum (m_E 0.135–0.141) have to cross it.

**Table P1. Predicted change of the floor best.** Previous best from `stage3/front_union_R2000.json`.

| λ0 | previous best | predicted new best | P(point > 0.02) | P(lower > 0.02) |
|---|---|---|---|---|
| none | +0.697 | +0.697 to +0.72; most likely unchanged or +0.00 to +0.01 | 0.15 | 0.05 |
| 0.3 | +0.426 | +0.426 to +0.52; median +0.44 | 0.35 | 0.25 |
| 0.5 | +0.383 | +0.383 to +0.45; median +0.39 | 0.35 | 0.20 |
| 0.7 | +0.262 | +0.262 to +0.36; median +0.28 | 0.50 | 0.35 |

**P1.1.** At least one floor has a point estimate above 0.02. I give this about 75%. The interval lower end is above 0.02 at one floor at most (about 55% for "exactly one or none", and I expect "none" slightly more often than "one").

**P1.2. Shape.** New front points appear in the m_E gap 0.15–0.42. They come from λ0 = 0.3 runs that end close to m_E = 0.30. Probability 0.6.

**P1.3. Floor-hugging endpoints.** Most feasible endpoints of the λ0 = 0.3/0.5/0.7 runs have m_E within 0.03 of their floor at R = 2000. The E-penalty (20 nat per unit) dominates the budget penalty, so the runs stop where the floor binds.

**P1.4. Start → endpoint.**
- *Low-stratum starts (m_E < 0.1) under λ0 ≥ 0.3.* These never become feasible, as in stages 2–3 (region-A runs stayed at m_E ≤ 0.048). Runs for different floors from the same low start follow identical paths whenever the floor is violated at every evaluation.
- *Feasible front starts (any floor).* The Δm_D change from start to endpoint is within ±0.03 nat. The median is +0.005, not resolved for most runs.
- *R = 200 pool starts.* At R = 2000 most of them are budget-qualified but violate the floor for λ0 ≥ 0.3, and all violate it for λ0 = 0.7 (LHS m_E ≤ 0.59). Infeasible → feasible transitions: about half of the λ0 = 0.3 runs, about a third of the λ0 = 0.5 runs, and at most 2 of the 6 λ0 = 0.7 runs.
- *R = 200 pool starts that are feasible.* Δm_D gains are large, +0.3 to +1.5 nat, because these starts are poor. None reaches the current front.

## 2. Script 1: the unbudgeted problem

**Basis.** The previous best is U-L1657, +2.101 (stage 3). The six best stage-3 endpoints lie within 0.04 nat (+2.062 to +2.101). They share p1 ≈ 0.42–0.45 and differ widely in phase, which indicates a broad plateau. The new starts are ranks 11–20 of the R = 200 Δm_D ranking (+1.25 to +1.35), lower than the stage-3 starts (+1.36 to +1.84). maxfev is raised from 200 to 300.

**P2.1.** The best new endpoints reach +2.00 to +2.12, all over budget (D ≈ +95 to +115 sends), with p1 in 0.40–0.46.

**P2.2. The unbudgeted best does not move by a resolved amount.**
- P(new best − U-L1657 > 0) = 0.4.
- P(point > 0.02) = 0.10.
- P(lower > 0.02) = 0.03.

## 3. Script 2: confirmation

**Pool size.** I expect 16–24 front points after Script 1, plus the 7 stage-2 finalists minus overlaps. NM-none-P5514, NM-0.3-P4312, AB-0.5 and UNIF may still be on the front. So n ≈ 20–26, M ≈ 228–375 and t ≈ 4.55–4.77.

**Expected precision** (scaled from the stage-2/3 batch se by √(R)):
- paired D: se ≈ 0.032 sends, corrected half-width ≈ 0.15 sends;
- m_E: se ≈ 0.0002–0.0007, half-width ≈ 0.001–0.003;
- Δm_D^(C) and pairwise differences: se ≈ 0.003–0.006, half-width ≈ 0.015–0.03 nat.

**P3.1. Budget.**
- *Undecided:* 2–6 of the n − 1 non-uniform candidates (point guess 4). Front points were selected at the budget boundary on the seed-20261012 draws. On independent draws some have true D within ±0.15 sends of 0.
- *Rejected:* 0–2, coming from Script-1 points selected on favourable draws.
- *Confirmed:* P5014 (D ≈ −3.5) and NM-none-P5450 (D ≈ −1.0), with near certainty.

**P3.2. m_E.**
- NM-0.7-UNIF (m_E 0.701–0.702 in stage 2) is undecided at λ0 = 0.7 (probability 0.85).
- Script-1 points that hug a floor (P1.3) add 1–3 further undecided (candidate, floor) pairs.
- S3-0.7-P4055 (0.705) is confirmed ≥ 0.7 (probability 0.7), unless it stops being the 0.7 incumbent.

**P3.3. Undecided feasibilities per floor** (undecided budget or m_E, neither rejected).

| | none | 0.3 | 0.5 | 0.7 | total over floors |
|---|---|---|---|---|---|
| point guess | 4 | 3 | 2 | 2 | 11 |
| range | 1–6 | 1–5 | 0–4 | 1–4 | 5–16 |

The total counts a budget-undecided candidate once per floor whose m_E it does not fail.

**P3.4. Incumbents.**
- λ0 = none: the incumbent is confirmed feasible (probability 0.7).
- λ0 = 0.7: the incumbent is confirmed feasible (probability 0.6).
- Candidate near-optimal sets: at λ0 = none, at least 3 members. The low-m_E cluster spans about 0.03 nat at R = 2000, below the exclusion distance δ + half-width ≈ 0.04–0.05. At λ0 = 0.7, 1–3 members.

## 4. Script 3: scenarios d = 5 and d = 50

The representatives are fixed by Script 1 (rule in `design.md` §3). I expect them to be:
- a high-D, low-m_E point (m_E ≈ 0.045, 1/m_E ≈ 22) for λ0 = none;
- points with m_E ≈ 0.30–0.42, ≈ 0.50–0.53 and ≈ 0.70–0.71 for λ0 = 0.3, 0.5 and 0.7, with 1/m_E ≈ 2.4–3.3, 1.9–2.0 and 1.4.

**Crude models (stated as models, not results).**
- *Information.* The D gain comes from the time distribution of the weights w_k on the same regressor geometry. Phase changes alone are a block rotation, so they leave det J unchanged at fixed weights (stage 1). At d = 50, non-event information is confined to a few σ = a few % of Δ before each event, so long event gaps are information-poor and clumpy sampling is penalised more. At d = 5 the band is only ±5σ wide, so long gaps still accumulate non-event information and the advantage of evenly spread events shrinks.
- *Sends.* The effective threshold per event is Δ + E[overshoot of z], with an overshoot of order σ. So B(σ) ≈ N^nf − kσ, with k depending on the input (slopes, turning points). For the uniform input N^nf = 321 and B_20 = 316.9.

**Table P4. Predicted direction of change relative to d = 20** (the same representatives, same quantities).

| quantity | d = 5 | d = 50 |
|---|---|---|
| Δm_D^(C), all four | sign stays positive; magnitude shrinks by 20–60% | sign stays positive; magnitude grows by 10–40% |
| 1/m_E, all four | smaller (closer to 1); high-D point 22 → 8–18 | larger; high-D point 22 → 25–40 |
| m_E vs 1 | stays below 1 for all four | stays below 1 for all four |
| order of magnitude of 1/m_E | 10¹ for the high-D point, 10⁰ for the three constrained points | the same |
| uniform budget B̂_d | 298–312 (below 316.9) | 318–321 |
| paired D of the representatives | D_5 − D_20 and D_50 − D_20 have opposite signs, and \|D_5 − D_20\| > \|D_50 − D_20\| (linear-in-σ send model) | (same statement) |
| budget feasibility (8.3.6(g)) | about 2 of 4 over or undecided | 1–2 of 4 over or undecided |

**P4.1. No representative is over budget at both d.** This follows from the opposite-sign statement and D_20 < 0. I give it 65%, since the linear-in-σ model is crude and N^nf is a non-monotone integer function of Δ.

**P4.2. Which representatives fail the budget at which d** is not predicted.

**P4.3. J_per(u_unif, N_d).** It has rank 9 at both d. The uniform input's SoD-vs-periodic offset, log det J_C − log det J_per, is +4 to +7 at d = 5 and +1 to +2.5 at d = 50 (it is +3.05 at d = 20).

**P4.4. Window input power.** It equals the d = 20 value exactly (deterministic; the consistency check).
