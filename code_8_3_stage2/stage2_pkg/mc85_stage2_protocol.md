# §8.5 stage 2 protocol and predictions (written before the joint search; 2026-10-10)

## Protocol (decided after review)
- Objective: D primary; report the empirical Pareto front (Delta m_D^(C) vs m_E^(C)) over all *evaluated* budget-qualified
  candidates. Local refinement runs several targets: unconstrained max D, and max D subject to m_E >= lambda_0 for
  lambda_0 in {0.3, 0.5, 0.7}; starting points taken from different regions of the front, not only the top-D points.
- Budget: hard constraint by 8.3.6(g) (qualified / over / undecided, uniform qualifies by definition) for the final tables;
  a penalty may steer the local search but never replaces the paired decision.
- Parametrisation (7-dim): p = softmax(z1,z2,z3,0), a_m = sqrt(p_m); chi in R^4 continuous inside the search, reduced
  mod 2*pi only when forming the input and when saving.
- Sampling of the first round: 3000 candidates, p ~ Dirichlet(1,1,1,1) (uniform on the simplex), chi ~ Uniform[0,2*pi)^4
  (LHS-stratified per coordinate), R=200 on the SAME noise draws as stage 1 (seed 20261010) for paired comparison;
  merged with the 1465 stage-1 candidates (deduplicated), uniform reference kept.
- Refinement: ~40 candidates (front ends, middle, budget-boundary) re-evaluated at R=2000 together with the reference;
  Nelder-Mead (penalty, maxfev=150 per start, R=500) from several front regions; every new candidate re-evaluated at
  R=2000 before any ranking. Estimates at different R are never mixed in one ranking.
- Final validation: 5-8 fixed candidates; batch 1 (independent seed) for J, paired send-count differences and design
  comparisons; batch 2 (another independent seed) for the ML covariance check of §8.4.
- Recorded per candidate: B_u, D +- se, decision, Delta m_D^(C), Delta m_D^(B), nine generalized eigenvalues vs (C),
  m_E^(B), mean gap and gap quantiles (clock steps), window mean AC power of input and output (1/T int u_ac^2 etc.),
  and for representative candidates: q_k = (1/R) sum_r eps_k^(r) (event rate per clock point) alongside x_k and w_k,
  and the worst-combination coefficients a = J_C v_min with v_min^T J_C v_min = 1, parameter order
  [G0, Re1, Im1, Re2, Im2, Re3, Im3, Re4, Im4].
- Wording: results are "the empirical Pareto front of the evaluated candidates", never a global optimum.

## Predictions (before running)
1. The joint search will find budget-qualified candidates with Delta m_D^(C) above the stage-1 phase-only best (+0.60),
   but by less than 0.5 nat: amplitude moves that raise log det also raise sends (stage 1), so within the budget the
   amplitude freedom buys little; most of any gain will again come from phases.
2. Along the empirical front, the maximal D is non-increasing in the m_E floor lambda_0 (feasible set shrinks) — this is
   a theorem for the true optimum and a consistency check for the search; we expect the front to be steep near
   m_E ~ 1 (small D gains cost much m_E) and flat for m_E < 0.1.
3. Not predicted: whether chi_3 - chi_4 keeps its stage-1 role once amplitudes vary; whether the best candidates move
   power away from uniform. To be read off the results.
4. The correlation between sends and log det across the LHS cloud will remain strongly positive (stage 1: 0.76 amplitude,
   0.91 phase) but with counter-examples, as in stage 1.
