# §8.5 stage 3: predictions (written before any stage-3 run; 2026-10-10)

These predictions were committed before the first stage-3 script ran. They rest only on stage-1/2 results already in `stage2/` and on the crude model below; no stage-3 number was computed.

Notation: Δm_D = Δm_D^(C) in nat, against the uniform reference on the same draws. "Half-width" = distance from the sweep centre (the stage-2 point) to where Δm_D first falls 0.05 nat below its centre value, taken on each side.

## Experiment 1: sharpness of the current points (R = 200 sweeps)

Basis: from P5514 to NM-none-P5514, Nelder–Mead moved the phases by only about (−5.5°, +5.8°, +2.9°, +1.0°). That move gained +0.120 nat at R = 2000, so the surface must change by tenths of a nat over a few degrees.

**P1.1, phase half-widths at NM-none-P5514.**
- **χ3 alone, χ4 alone, and χ3 − χ4 (sum fixed):** 5°–20°.
- **Ordering:** the difference sweep (c) is the narrowest of the three, because the close pair's relative phase sets where the beat envelope sits in the window.
- **χ1 and χ2 alone:** 10°–30°.
- **Against the stage-1 step:** every phase half-width is well below the 60° step, a ratio of about 0.1–0.35.

**P1.2, p1 sweep at NM-none-P5514.**
- **Shape:** Δm_D *increases* with p1 over most of [0.15, 0.40], with no interior maximum near p1 = 0.25. This follows from stage 1, where p = (0.4, 0.2, 0.2, 0.2) at χ = 0 gave +0.61 while over budget.
- **Upper side:** probably no −0.05 crossing.
- **Lower side:** the crossing lies within 0.01–0.02 of the centre (p1 = 0.252).
- **Budget:** D rises by roughly 2–4 sends per 0.01 in p1. The point turns over budget (or undecided at R = 200) immediately above its centre p1, so the budget, not the information, sets p1 there.

**P1.3, AB-0.5.** The same orders of magnitude.
- **Phase half-widths:** somewhat wider, 10°–30°, since its region of the landscape was flatter in stage 2.
- **p1 sweep:** again increasing in p1 and budget-limited at its centre (p1 = 0.209).

**P1.4.** Both stage-1 grid steps were too coarse to resolve the peaks: 60° in phase, and 0.1 (coarse) or 0.05 (near-uniform) in p.

## Experiment 2: unbudgeted (power-only) problem

**Crude model, stated as a model and not a result.** Assume N ∝ rms slope of x, with s_m = |G(jω_m)|ω_m = 0.445, 0.21, 0.168, 0.16, and log det ≈ 9 log N + 2 Σ log p_m. Maximising over the simplex gives p ≈ (0.46, 0.19, 0.18, 0.17) and about +1.0 nat over uniform at χ = 0.

**P2.1, power split.** The best evaluated point of the unbudgeted problem has p1 between 0.40 and 0.55, most likely about 0.45–0.48.
- **p1 > 0.5?** I predict **no**, with modest confidence (about 60%).
- **Other tones:** they keep 0.15–0.22 each. None is driven towards zero, because the 2 Σ log p_m term penalises that strongly.

**P2.2, value and sends.** Its Δm_D at R = 2000 is +1.8 to +2.6 nat. Its send count B is about 380–460, i.e. +20% to +45% over the budget, so D is strongly positive and the decision is "over".

**P2.3, SoD vs matched periodic.** The advantage log det J_u − log det J_per(u, ⌊B_u⌋) for the best point and the five best endpoints is +2.0 to +3.3 nat.
- It is similar to or smaller than the uniform input's value at its own budget (+3.06 at R = 2000).
- The reason: at a higher event rate, the non-event information per event should not grow.

**P2.4, conditioning.** The nine generalized eigenvalues vs (C) of the best point are mostly above 1 (more information overall). The smallest is below 0.5, i.e. some combination still loses information against uniform.

## Experiment 3: denser budgeted search

**P3.1, grids.** Neither 30°-step phase grid (625 points around NM-none-P5514 and around AB-0.5) contains a point clearly better than its centre. The 30° step is wider than the predicted half-widths, so the grid mostly samples the flanks.
- **At R = 200:** the best qualified grid point is within about ±0.05 nat of the centre's R = 200 value.

**P3.2, unconstrained floor (λ0 = none).** The best qualified Δm_D at R = 2000 improves on +0.689 by **0 to +0.06 nat** (median guess +0.02).
- I put about even odds on the gain exceeding 2 paired se.

**P3.3, m_E floors.**
- **λ0 = 0.3 and 0.5:** the front moves more, by +0.02 to +0.15 nat. These floors had fewer evaluations; the λ0 = 0.5 own-search outcome was only +0.240 against the shared-pool +0.383.
- **λ0 = 0.7:** at most +0.03, and probably not resolved.

**P3.4, monotonicity.** Both the shared-pool best and the own-search outcomes stay non-increasing in λ0.

**P3.5, budget margin.** All new qualified best points still sit at the budget boundary, with margin −(D + 2se) below 0.5 sends at R = 2000.
