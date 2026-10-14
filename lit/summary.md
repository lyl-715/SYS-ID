# Literature summary: input design for event-triggered / quantized / censored identification

Based on `comparison.md` (13 items: 8 read in full and 5 from the abstract only). Statements about the
abstract-only items are limited to what their abstracts say. Where a point is my own derivation from a
paper's result, rather than something the paper states, it is marked **(inference)**. Nothing is quoted.

## 1. Global guarantees and the structure that makes them possible

| Paper | What is globally optimal | Structural assumption that makes it possible |
|---|---|---|
| Casini et al., TAC 2011 | N-step worst-case optimal input for a **scalar gain**, under both the absolute-error and the relative-error criterion; the second is recursive (does not depend on N) | One parameter with an interval prior, bounded noise, one threshold. The feasible set stays an interval, and each reading cuts it at a point the input chooses. With an adversarial outcome, the best cut makes both outcomes equally bad, and induction closes the recursion. The input is fed back from past readings |
| Casini et al., CDC 2008 | one-step optimal input for a scalar gain, any thresholds, with noise | same interval structure; the one-step min-max is piecewise linear over 2P−1 intervals |
| Casini et al., Automatica 2012 | one-step optimum (generic and equispaced thresholds); **N-step** optimum for noise-free equispaced thresholds | as above, plus a range/resolution condition (lower/upper prior bound ≥ (C_1−Q)/(C_P+Q)) under which the problem reduces to the binary case after the first step |
| Molin & Hirche, 2012 | joint trigger and estimator for remote state estimation: a symmetric threshold plus a linear predictor | first-order scalar system; symmetric **and** unimodal initial-state and noise densities. The alternating algorithm then converges to that pair from any start, consistent with Lipsa & Martins (2011). Symmetry alone is not enough: a bimodal example beats it by up to ≈45% |
| Dozzi et al., 2026 | the D-optimal **sampling density** | the regression is linear in the parameters, regressors are fixed by a known input, and sampling is exogenous (independent of noise and output). The expected FIM is then linear in the density, so log det is concave over a convex set. The implementable deterministic sequence is greedy (no guarantee), and the criterion is the expected FIM, not the expected covariance |
| Godoy et al., 2014 (abstract only) | search summaries of the abstract mention a convex problem via the graph-theory method | if correct: global within a finite-alphabet stationary input class, at a nominal θ (local in θ). Unverified |

No global guarantee:

- Trimpe & Campi: one-step (myopic) trigger rule with a Kalman filter that ignores non-events.
- González et al. and Kawaguchi et al.: estimators only (EM / BFGS, local).
- Diao et al. and Guo et al.: not visible from the abstracts.
- Every FIR result in the Casini line is suboptimal: coefficients are excited one at a time, and whether that is
  optimal is listed as open.

**Common thread.** Global results come from one of three structures:

- **(i)** a scalar unknown whose uncertainty set stays an interval;
- **(ii)** symmetry plus unimodality, which makes threshold policies optimal;
- **(iii)** information that is linear in a freely chosen design measure.

**Fit to SoD identification.** None holds by default:

- θ has 2 (CLAUDE.md model) or 9 (the §8 multisine) components.
- The sampling density is induced by the output and depends nonlinearly on θ and on the input.
- With a noisy trigger, the law of the censoring band shifts with the history.

**Partial fit (inference).** (iii) holds if one conditions on event times and treats the induced phase density as
the design object. The repo's ratio det I_ev / det I_per = 1 − |ζ|² is then Dozzi's argument (§4b). But the input
is what can actually be designed, and it reaches the density only through the trigger, so convexity is lost.

## 2. Does any paper handle a trigger whose threshold depends on the last sent value?

**Among identification papers: no.**

The closest are two state-estimation papers:

- **Trimpe & Campi** include SoD explicitly (send when the measurement moves δ from the last sent one). Their
  Kalman filter treats the send pattern as a fixed schedule and discards what a non-send says. SoD is only
  compared by Monte Carlo, where it is the weakest trigger and fails for an unstable process.
- **Molin & Hirche** trigger on the deviation of the state from the prediction made from the last sent value; for
  a = 1 this is SoD on the state, with thresholds from dynamic programming. Their estimator uses the censoring
  exactly: its bias term is the conditional mean of the accumulated noise given no send since the last one. For
  symmetric unimodal noise that term is zero, so the optimal estimator ignores the censoring. For bimodal noise it
  does not, and exploiting the censoring then pays.

The level-crossing identification papers (Kawaguchi et al., González et al.) use a **fixed** amplitude grid with
no hysteresis:

- In Kawaguchi, a non-send at clock step k means the output is still in the band of step k−1. The band is indexed
  by the history in form. In content it is the same as reading a memoryless quantizer at every step
  **(inference)**.
- González states this directly: the band is defined at every clock step.

**What is different under SoD.**

- The band (r−Δ, r+Δ) is centred on the last sent value r, so it is a hysteresis quantizer whose state is r, with
  width 2Δ rather than one grid step.
- When the noise is white on the clock, the likelihood still factorises: r is a function of past observed data,
  so Kawaguchi's derivation goes through with the band replaced **(inference)**.
- What does not carry over is the closed-form average over the band. Kawaguchi averages over the previous band,
  whose probability is just the marginal at k−1. Under SoD the law of r depends on the whole noise path and needs
  Monte Carlo or a forward recursion over r. The repo's §8 code already does the Monte Carlo version: `cfun`
  weights are averaged over realisations.

**Other items:**

- Diao et al. (2018) and Guo et al. (2024): the trigger rule is not in the abstracts, so this is unknown.
- Casini et al.: the history dependence is in the **input** (sequential design), not in the sensor.

## 3. Is any Fisher-based design validated by actual estimation runs?

- **Dozzi et al.: yes, the only one.**
  - The D-optimal sampling sequence is compared with uniform sampling over 500 least-squares runs per N.
  - The trace of the sample covariance is lower, and about half the samples give the same accuracy.
  - Caveats:
    - With exogenous sampling and LS, σ²(ΦᴴΦ)⁻¹ is the exact covariance given the sampling times, so the Monte
      Carlo confirms algebra rather than testing an asymptotic approximation.
    - One system, one input, one sequence; runs vary only the noise.
- **Kawaguchi et al.** validate a Fisher-based **analysis**, not a design.
  - 1000 ML runs per spacing reproduce the inverse FIM and the predicted send count across level spacings.
  - The output-error estimate from sent samples alone is biased.
  - Their broad Lebesgue-vs-periodic comparison (2000 random systems) uses the FIM only.
- **Godoy et al. (abstract only):** a numerical example; whether it includes estimation runs is not visible.
- **No Fisher-based design here:**
  - Casini et al.: worst-case sets computed with adversarial noise; a random-input baseline averaged over input
    draws.
  - Trimpe & Campi: Monte Carlo of the estimation error, but the trigger design is not Fisher-based.
  - González et al.: Monte Carlo of an estimator.
  - Molin & Hirche: criterion value only.

**Gap.** No paper validates a Fisher-based **design** in a censored or event-triggered setting, where the FIM is
only an asymptotic stand-in for the covariance. The repo's MLE-vs-CRLB check on the censored likelihood
(`mc84.py`, §8.4) is the kind of evidence the literature lacks.

## 4. Upper bounds on information under a communication constraint that could be adapted

None of the papers states such a bound for event-triggered identification. Four pieces can be adapted:

**(a) Clock-rate ceiling, from Kawaguchi's FIM (inference).**

- Each clock step contributes h_k φ_kφ_kᵀ, with 0 ≤ σ²h_k ≤ 1. The upper limit holds because coarsening a
  Gaussian observation cannot add Fisher information (data processing). The same holds for SoD, i.e. for the
  repo's c_ev + c_no ≤ 1, since the band is fixed by past data.
- Hence I_event ⪯ I_clock = (1/σ²) Σ_k φ_kφ_kᵀ over all clock steps.
- With N̄ expected sends out of N_clock steps, and regressors that are roughly stationary over the window,
  approximately I_event ⪯ (N_clock/N̄) I_per(N̄).
- So no SoD scheme can gain more than the ratio of clock rate to send rate over periodic sampling at equal count.
  In the repo's setting (5000 steps, about 316 sends) the factor is about 16.
- This is loose when Δ ≫ σ (the repo uses σ = Δ/20), because then c_no ≈ 0 except near band edges.

**(b) Hadamard / leakage bound, from Dozzi et al.**

- If information comes only from the N sent samples, and the noise is independent of the sampling times (event
  times treated as given, as in the CLAUDE.md model), then at fixed N:
  det I ≤ product of the diagonal entries, which N and the input amplitudes fix.
- Equality holds iff the off-diagonal (leakage) terms vanish. For a single sine this is det I_ev ≤ det I_per, with
  ratio 1 − |ζ|², the repo's established result.
- Two consequences **(inference)**:
  1. Under that model, SoD cannot beat leakage-free sampling at equal N in the D-criterion. Any event-based gain
     must come from the censoring information in (a), and including it is a modelling choice.
  2. For the repo's nonperiodic multisine (tones 12 and 12.3 rad/s in a 5 s window), uniform sampling is itself
     not leakage-free. A sampling pattern, induced or designed, can therefore beat uniform sampling at equal N
     even without censoring information.

**(c) Per-reading worst-case cap, from Casini et al. (CDC 2008, Automatica 2012).**

- One P-threshold reading can shrink the worst-case interval by at most a factor 1/(P+1). Noise-free, this gives a
  time complexity of at least log_{P+1}(D_0/D_N), i.e. at most log2(P+1) bits per reading.
- An SoD event carries a direction bit plus a crossing time. Adapting the cap would require bounding the
  information in the timing, which no paper read here does.

**(d) Molin & Hirche: rate-penalised trade-off curve.**

- Their cost (mean squared error plus λ times sends) gives the best achievable error-vs-rate trade-off for state
  estimation.
- This is an achievability curve, not an information bound. Using it for parameters would need a Fisher-type
  stage cost, which they do not treat.

**Pointer, not read:** Wang & Yin, IEEE TAC 55(3), 2010 (cited in Casini et al. 2012). It defines a Fisher
information ratio for quantized communication channels and is the most likely source of a directly adaptable
information-ratio bound.

## 5. Input or experiment design specifically for Lebesgue / SoD-sampled identification

**None found.** No paper located designs the input (or the threshold) for identification from send-on-delta or
Lebesgue-sampled data. The closest items:

- **Guo et al. (TCAS-II, 2024, abstract only):** input design for FIR identification with binary-valued outputs
  under event-triggered communication. This is 1-bit and discrete-time, not SoD on a continuous-valued output.
- **Dozzi et al. (2026):** design of the sampling times themselves, which are chosen, not output-triggered.
- **Kawaguchi et al. (2017):** Fisher analysis under Lebesgue sampling. They name the effect of the input
  sequence, and the choice of thresholds, as future work.
- **González et al. (2023/2024):** estimation only, no design.

**Searches run.**

- 12 web searches; the queries and closest hits are logged in `sources.md`.
- Reference lists of Dozzi et al., González et al. and Kawaguchi et al., checked by hand.
- Related works found and not read:
  - Kawaguchi et al., CDC 2016 (English companion);
  - Kawaguchi et al., CCTA 2017;
  - González et al., arXiv 2304.03312 (IFAC 2023 version) and 2304.03314 (EM, state-space);
  - Pouliquen et al., CDC 2016 (continuous time, binary measurements);
  - Guo et al., IEEE Access 2018 (scheduled, not event-triggered, binary observations; its abstract mentions
    input design).

  None of these is input design for Lebesgue / SoD sampling, as far as their titles and abstracts show.

**Limits of this search.** Bibliographic databases (IEEE Xplore, Scopus, Google Scholar, Semantic Scholar) were not
reachable from this session, so the negative result rests on web search plus three reference lists.
