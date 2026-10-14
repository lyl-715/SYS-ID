# Route 1: review note (2026-10-14)

This note records the review changes to the task. They arrived after `predictions.md` was committed (76ff59b) and before
any bound script had produced a committed result. Each change was adopted as follows.

## (1) Outer search dropped (Script 3 replaced)

**Review.** For any fixed weight vector, H(u) = H(u_unif) T(u). Here T is block-diagonal, with an amplitude scaling times a 2-D
rotation for each tone, and det T = ∏_m 4p_m. So for every fixed W, U_W(u) = U_W(u_unif) + 2 Σ_m log(4p_m). This is
independent of the phases and maximised at uniform power. The outer search therefore adds nothing.

**Implemented.**
- `bound_outer.py` was never written. The multistart Nelder–Mead (≥ 200 starts, seed 20261022, maxfev 400) was not run.
- It is replaced by `bound_identity.py`, which states the identity and checks it numerically:
  - the factorisation H(u) = H(u_unif) T(u) and det T = ∏ 4p_m, at 20 random inputs and the 7 finalists;
  - the log-det residual F_u(ξ) − F_unif(ξ) − 2 Σ log(4p_m) at random *fixed* feasible ξ, with non-uniform p, non-zero
    phases, capped and uncapped weights;
  - the identity at the optimum: U_316(u) solved directly, cold, at 4 random inputs and the 7 finalists, against the
    prediction.
- The random numbers (p ~ Dirichlet(1,1,1,1), χ ~ U[0, 2π)⁴, random ξ) use seed 20261022, the seed fixed for Script 3's
  multistart. They are the only random numbers of Route 1.
- The same identity had already been derived in `predictions.md` §0, before any run. Its outer-problem predictions (§3) are
  now answered by the identity, not by a search.

## (2) Variant B uses each candidate's own S(u)

**Review.** B must use the candidate's own S(u), not 466.

**Implemented.**
- **Uniform input (Script 1).** W_B = S(u_unif) = Σ_k ŵ_k = 465.958 (`wk_d20_10000.npy`). The value at W = 466 is printed
  once as a note and not used.
- **Finalists (Script 3).** S(u) = Σ_k ŵ_k comes from the batch-1 weights (`stage2/final_qxw.npz`, R = 4000, seed 20261013).
  The relaxation is solved at W = S(u) and input u.
- **Reported quantity.** ld J_sod(u) − U_{S(u)}(u), called the **free-placement relaxation gap at equal total weight**.
- **Wording.** 466, or S(u_unif), is not presented as a bound for other inputs.

## (3) Certified inner solves

**Review.** Report F(ξ̃) and the cutting-plane upper bound F(ξ̃) + max_{ξ∈Ξ_W} ∇F(ξ̃)ᵀ(ξ − ξ̃), with its gap. Only the upper value
is quoted as the bound.

**Implemented** (`bound_common.certificate` / `certificate_full`).
- **The bound.** ∇F(ξ̃)_k = d_k = h_kᵀ M⁻¹ h_k. The linear program over the box-and-sum set Ξ_W is solved by sorting: the
  ⌊W⌋ largest d_k at the cap, plus the remainder on the next one. With no cap it is W·max d.
- **What is printed.** Every solve reports F, U_cp = F + top-sum(d) − ξ̃ᵀd, and gap_cp. This covers the exchange method,
  SPG, Clarabel and SCS (after projection onto Ξ_W), the brentq solves, the Dozzi density and the finalists.
- **Stopping rule.** Both own solvers now stop on the cutting-plane gap.
- **Duality bound.** F + 9 log(top-sum/9) is printed as a secondary, tighter check.
- **What the tables quote.** U_cp.

## (4) Wording

**Review.** (A) is "the relaxation bound for 316 full measurements on the clock grid, each instant at most once", not a
distance of send-on-delta from its own limit. Dozzi et al. optimise a continuous sampling distribution and construct a grid
sequence afterwards, so the discrete capped problem is related to theirs, not identical with it.

**Implemented** in the script docstrings, the log statements, the figure labels and `bound_report.md`.
- `bound_dozzi.py` states the relation explicitly.
- It adds a cutting-plane bound for the *continuous* density problem: the gradient d(t) is evaluated on a 0.01 ms grid
  over [0, T]. This shows how far the continuous optimum can lie above the clock-grid version.
