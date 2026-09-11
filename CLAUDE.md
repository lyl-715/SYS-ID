# Project

Internship on input design for event-based system identification
(TU/e, Control Systems Technology).

## Model

    y(t_k) = phi(t_k)^T theta + e(t_k),   e ~ N(0, sigma^2) iid
    phi    = [u, udot]^T,  theta = [theta1, theta2]^T
    I      = (1/sigma^2) sum_k phi(t_k) phi(t_k)^T

Send-on-delta triggering with a FIXED threshold (not a design variable):

    tau_{i+1} = inf{ t > tau_i : |x(t) - x(tau_i)| >= Delta },  x = phi^T theta

## Existing code

- `event_input_design.py` — single-sine experiments, periodic vs event-based
  sampling, the Fisher information matrix and D/A/E criteria. Two model
  structures: `MODEL_STATIC` (G(p) = theta1 + theta2 p) and `MODEL_DYNAMIC`
  (two first-order lags).
- `noisy_trigger.py` — triggering on the noisy output, band-limited noise,
  Rice-formula prediction of the event rate.

Reuse the functions in these files. Do not rewrite them.

## Established results

The induced sampling density in output phase is `p(s) = |cos s| / 4`.
Define `zeta = <exp(2i s)>`, the second harmonic moment. With
`q = zeta exp(-2i psi)`:

    det I_ev / det I_per = 1 - |zeta|^2
    lambda_min ratio     = 1 - |zeta|
    condition number     = (1 + |zeta|) / (1 - |zeta|)
    I22 / I11            = w^2 (1 + |zeta| cos 2psi) / (1 - |zeta| cos 2psi)

For a single sinusoid `|zeta| = 1/3`, measured as invariant across 36
combinations of (A, w, theta) and two model structures, within a
finite-threshold bias `|zeta| ~ 1/3 + (2/3)(Delta/C)`. Periodic sampling is
`zeta = 0`. Triggering on a noisy output drives `|zeta|` toward 0.

## Environment

Cloud session: 4 vCPUs, 16 GB RAM, no shell for the user. Use
`multiprocessing` with 4 workers for any sweep over more than ~100 cases.
Print results as text tables as well as saving figures, and commit figures
to the repo so they can be viewed on GitHub.

## Conventions

- Exact crossing detection with linear interpolation; events must not land
  on grid points.
- After triggering, set the reference to the exact target level
  `ref + sign*Delta`, not to the grid sample value.
- Compare sampling schemes at equal event count N.
- Use steady-state analytic expressions, not an ODE solver.
- Where a result depends on a modelling choice, say so rather than
  presenting it as settled.
