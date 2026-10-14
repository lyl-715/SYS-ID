# Route 1, free-weight D-optimal relaxation: predictions (written before any bound run; 2026-10-14)

I committed these predictions before the first bound script ran.
Before writing them, I computed only two quantities, both already in the repository:
- log det J_sod(u_unif) = 197.197, from `wk_d20_10000.npy`, with Σ ŵ_k = 465.958;
- the largest single weight, max ŵ_k = 1.000 (at k = 0).

No relaxation was solved. No Dozzi sequence was built.

Notation:
- ld(·) is log det in nat.
- J_per = J_per(316): periodic, N = 316, uniform input, ld = 194.155.
- J_sod = J_sod(u_unif): ld = 197.197, total weight S = 465.96.
- A, B and C are the three relaxation variants of the prompt. W is the total weight, and "cap" means ξ_k ≤ 1.
- The geometric-mean axis ratio of the 9-parameter confidence ellipsoids is e^{−Δ/18}, where Δ = ld(1) − ld(2).

## 0. A structural fact used below (analytic, not a run)

The regressor of `regressors(a, chi, t)` (mc85_scan.py) factorises as h_k(u) = T(u)ᵀ g_k, where:
- g_k = [1, cos ω_m t_k, sin ω_m t_k]_{m=1..4} does not depend on the input;
- T(u) is block-diagonal: a0 = 1, then one 2×2 block per tone, a_m·[[cos χ_m, −sin χ_m], [−sin χ_m, −cos χ_m]], with |det| = a_m².

So for **every** weight vector ξ,

    log det( σ⁻² Σ_k ξ_k h_k(u) h_k(u)ᵀ ) = Σ_m 2 log(4 p_m) + log det( σ⁻² Σ_k ξ_k h_k(u_unif) h_k(u_unif)ᵀ ),   p_m = a_m².

Consequences:
- The inner optimiser ξ* is the same for every input. Only the inner value shifts, by the constant 2 Σ_m log(4 p_m).
- The phases χ do not enter the inner value at all.
- 2 Σ_m log(4 p_m) ≤ 0 by Jensen (Σ p_m = 1), with equality only at p_m = 1/4.

This is Dozzi et al.'s factorisation Φ = Ψ A, which removes the input from det.

## 1. Inner problem at the uniform input (Script 1)

The heuristic behind the numbers is one effect.

**Close pair.** The pair ω3 = 12, ω4 = 12.3 is the only strongly leaking pair: Δω·T = 1.5 rad.
- With fast phases averaged out, the 4×4 block of that pair has det ∝ (1 − |ζ|²)², where ζ = E_ξ[e^{j0.3t}].
- For uniform sampling, |ζ| = 0.909.
- For two equal end clusters, |ζ| = cos 0.75 = 0.732, which is the minimum on [0, 5].
- So the most that moving weight to the window ends can gain is 2 log(0.465/0.174) ≈ 1.96 nat.

**Other pairs.** The other pairs have |ζ| ≲ 0.11, so less than about 0.03 nat each is available there.

**Effect of the cap.** At W = 316 the cap ξ_k ≤ 1 forces the end clusters to span at least 0.16 s each. The 12 rad/s tones then need phase coverage inside the clusters. Both effects reduce the gain.

| quantity | prediction | range |
|---|---|---|
| ld(A) − ld(J_per) | **+1.5 nat** (axis ratio 0.92) | +0.9 to +2.2 |
| ld(A) − ld(J_sod) | **−1.55 nat**: J_sod lies *above* the full-sample schedule limit at 316 | −2.3 to −0.9 |
| ld(B) − ld(J_sod) | **+1.7 nat**: **more** than 1 nat (P ≈ 0.75) | +1.0 to +2.5 |
| ld(C) − ld(A) | **+0.7 nat** | +0.3 to +1.5 |
| ld(C) − ld(J_per) | +2.2 nat | +1.5 to +3.0 |

**Why J_sod sits above A.** SoD collects weight 466 from 316 sends, because the non-send intervals carry censoring information.
- At equal placement, 466 instead of 316 is worth 9·log(466/316) = 3.50 nat.
- SoD realises J_sod − J_per = 3.04 nat of this. So its placement, with weight ∝ |ẋ|, costs about 0.46 nat relative to uniform placement at the same total. This is derived from existing numbers.
- Bound B should recover that 0.46 nat plus the end-cluster gain at W = 466. I expect the cluster gain at W = 466 to be smaller than at 316, about 1.2 nat, because 9.3% of the grid must now carry weight.

**Shape of the optimal weights.**
- (A) and (B) are bang-bang: ξ_k ∈ {0, 1} except a few interior weights. I predict at most 10 interior weights per variant. KKT requires d_k = ν exactly for an interior weight, which on a discrete grid happens only at isolated crossings of the level ν.
- The ones lie in two blocks at the window ends, plus a few short blocks inside.
- (C) puts its mass near both ends on 15–30 support points, below the Carathéodory limit of 45.

**Solvers.**
- cvxpy/Clarabel and my own projected-gradient method will agree to better than 1e-5 nat on (A).
- The certified duality gap of the projected-gradient solution, 9·log(Σ_top-W d_k / 9), will be below 1e-6 nat.

**J_per off the grid.** J_per uses t_i = 5i/316. These times are not on the 1 ms clock grid, so strictly J_per is not a feasible point of (A). The grid-rounded periodic schedule differs from it by less than 0.01 nat.

## 2. Dozzi design (Script 2)

The paper specifies the D-optimal density (eq. 15) and a greedy sequence (§4.1). Two things are left open, and I will fix them by stated choices:
- the grid size L;
- the start, since R_k is singular for k < 9.

I implement **without replacement** on the clock grid. The paper's remark that the grid resolution bounds the inter-sample time implies this.

Predictions:
- The greedy sequence lies between J_per and A.
- ld(A) − ld(greedy) = 0.1–0.5 nat, so greedy recovers most of the gain.
- The start choice moves its log det by less than 0.1 nat.

**Density versus sequence.** On the clock grid the density of eq. (15) is variant C normalised. Its expected FIM at N = 316 therefore equals bound C. It is not realisable with 316 distinct full samples.

## 3. Outer problem over the input (Script 3)

This follows from §0 and is exact, not a guess.

| question | prediction |
|---|---|
| does the optimum move power away from uniform? | **No.** p = (1/4, 1/4, 1/4, 1/4) is the global maximiser of the bound, for A and for B |
| does it change phases? | **No.** The bound does not depend on χ |
| gain of the outer optimum over the uniform-input bound | **0 nat exactly**. Numerically the best evaluated point will match it within the solver tolerance, at about 1e-6 nat |
| best found at uniform power with free phases − at zero phases | 0 to solver tolerance |

**Nelder–Mead behaviour.** NM will converge in z to z ≈ 0, which is p uniform. It will drift arbitrarily in χ, because that direction is flat.

**Warm start.** Since ξ* does not depend on u, a warm-started inner solve needs only a few iterations. I expect under 10 ms per warm solve and under 2 s per cold solve.

**Stage-2 finalists against bound B.** For these inputs, J_sod(u) should sit below bound B evaluated at W = S(u) (their own total weight):
- by about 1.7–1.8 nat for UNIF;
- by about 1.0–1.1 nat for NM-none-P5514, whose J_sod is 0.697 higher while its bound is unchanged up to 2Σlog(4p_m) ≈ −0.001.

**What input design can add at most.** Given the trigger, the free-weight bound says it can add nothing beyond what moving the weights could do. The +0.697 nat of the best budgeted input is entirely a better placement of the SoD weights. It is not an increase of the bound.

## 4. Runtime

- Script 1 (Clarabel at K = 5000, three variants, plus projected gradient): under 5 min.
- Script 2: under 1 min.
- Script 3 (≥ 208 starts × 400 evaluations × 2 variants ≈ 170 000 warm inner solves on 4 workers): under 15 min.
