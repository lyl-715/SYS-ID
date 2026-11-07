# Round 11b (addendum) predictions — committed before any computation

Branch `round11-edge-actions`, parent `395a11a`. At the time of this commit no round-11b code exists and no round-11b number
has been computed (only archived JSON fields of rounds 7 and 11 were read).

Notation: S = the round-11 discrete edge SoD class (E, 10 values, 2σ minimum distance from r); S′ = the class of
DECISION 11b-1 (E′, 28 values, 1σ minimum distance from r); T₁ = `round11/results/table_edge_iter.npz`,
I(T₁) = 207.35521730257756; B = 316.8921168493721; B′ = B + 0.1; p = 9.

## Predictions (prompt §5, verbatim)

```
- UB_SoD(S′) − I(T₁) ∈ [0.06, 0.40] nat
- the B′ = B + 0.1 correction adds < 0.005 nat
- the primal table T′ improves on I(T₁) by < 0.20 nat (so the certificate gap, whichever table, stays < 0.40)
- argmin λ is at or next to the final bisection step, as in Round 11
Scoring: hit iff the reported value lies in the stated range.
```

## Scoring rules (fixed now)

1. Value = UB_SoD(S′) − I(T₁) at budget B (not B′). Hit iff 0.06 ≤ value ≤ 0.40.
2. Value = UB_SoD(S′; B′) − UB_SoD(S′; B). Hit iff value < 0.005.
3. Value = I(T′) − I(T₁) on σ/24 (negative values count as < 0.20). Hit iff value < 0.20. The bracketed consequence is
   scored separately: hit iff max(UB − I(T₁), UB − I(T′)) < 0.40, with UB the B-version.
4. Value = run number of the λ that attains min_λ D(λ) in the S′ scan. Hit iff it is the last run of the scan or the run
   before it, and that run is a bisection step.

## Notes recorded before running (to be reported, not acted on silently)

- **S ⊄ S′ as defined.** Under DECISION 11b-1 the r-side edge is r ∓ 1σ. The S windows whose edge is the 2σ r-side clamp
  (r ∓ 2σ, active whenever x̄_k ∓ e σ lies inside r ∓ 2σ, e.g. the far side when |r − x̄_k| > e σ) are in S′ only if some
  e′ ∈ E′ puts the x̄-based edge on exactly that node. So "the bound over S′ ≥ the bound over S by construction" does not
  hold in general, and T₁ (which uses the 2σ clamp) need not belong to S′. Proposition 5 (tangent plane at J(T₁)) does not
  need T₁ ∈ S′ and stays valid for every T ∈ S′. Check D (min D(S′) ≥ min D(S) − 1e-9) and check 5 are therefore
  numerical checks, not consequences of an inclusion. The report quantifies the mass of T₁ on windows that S′ cannot
  represent, and compares V_λ(S′) with V_λ(S) at identical λ.
- **The B′ = B + 0.1 premise.** `round7/results/partC_regrade.json` gives max d1 = E[N]₂₄ − E[N]₁₂₀ = 0.078 (old action
  set), but `round11/results/round11.json` gives E[N]₂₄ − E[N]₁₂₀ = 0.197 (edge_j0), 0.210 (edge_j1), 0.206 (edge_j2) for
  the edge tables. B′ = B + 0.1 therefore does not cover every edge-class table with σ/120 E[N] ≤ B. The B′ bound is
  computed as specified; the report also gives UB(B + δ) = I(T₁) − p + min_λ [D(λ) + λ δ] for larger δ (no new DP).
