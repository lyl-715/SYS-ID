"""
Triggering on the NOISY output  (Caveat 1 of the progress note).

    tau_{i+1} = inf{ t > tau_i : |y(t) - y(tau_i)| >= Delta },   y = x + e

Noise model
-----------
e(t) must have C^1 paths for Rice's formula (Azais-Wschebor Thm 3.2) to
apply -- and, more basically, because continuous-time white noise would
make the trigger fire immediately.  We therefore use band-limited noise,

    e(t) = sum_k a_k sin(w_k t + phi_k),   w_k uniform on (0, w_c],
           phi_k ~ U[0,2pi)

which is smooth, Gaussian-ish by CLT, and has independently controllable

    sigma_e^2  = sum a_k^2 / 2
    sigma_de^2 = sum a_k^2 w_k^2 / 2      (flat spectrum -> = sigma_e^2 w_c^2/3)

Rice prediction
---------------
For a stationary Gaussian e, y(t) and ydot(t) are independent at equal t,
so the conditioning in Thm 3.2 drops out and the crossing intensity of
level u is  E|xdot + edot| * phi_sigma(u - x(t)).  Summing over the grid
{j*Delta} and replacing the sum by an integral of spacing Delta,

    lambda(t) = E|xdot(t) + edot(t)| / Delta                        (*)

with the folded-normal mean
    E|a + Z| = s*sqrt(2/pi)*exp(-a^2/(2 s^2)) + a*erf(a/(s*sqrt2)),
    Z ~ N(0, s^2),  s = sigma_de.

Note sigma_e does NOT appear in (*): the event rate is set by the
variance of the noise DERIVATIVE, not of the noise.

Caveat on (*): Rice counts crossings of the fixed grid.  Send-on-delta
does not re-trigger on the level it is currently sitting at, so (*) is an
UPPER bound, tight only while the drift dominates the noise.  Quantified
below.
"""

import numpy as np
from scipy.special import erf

from event_input_design import output_envelope, periodic_instants, fisher, criteria


# --------------------------------------------------------------------------
# Band-limited noise
# --------------------------------------------------------------------------

def make_noise(sigma, w_c, n_comp=200, rng=None):
    """Return callables e(t), edot(t) and the exact sigma_edot."""
    rng = rng or np.random.default_rng(0)
    w = rng.uniform(1e-3, w_c, n_comp)
    phi = rng.uniform(0, 2 * np.pi, n_comp)
    a = np.full(n_comp, sigma * np.sqrt(2.0 / n_comp))     # sum a^2/2 = sigma^2
    sigma_de = np.sqrt(np.sum(a ** 2 * w ** 2) / 2)

    def e(t):
        return (a[:, None] * np.sin(np.outer(w, t) + phi[:, None])).sum(0)

    def edot(t):
        return (a[:, None] * w[:, None]
                * np.cos(np.outer(w, t) + phi[:, None])).sum(0)

    return e, edot, sigma_de


def folded_normal_mean(a, s):
    """E|a + Z|, Z ~ N(0, s^2)."""
    if s <= 0:
        return np.abs(a)
    return (s * np.sqrt(2 / np.pi) * np.exp(-a ** 2 / (2 * s ** 2))
            + a * erf(a / (s * np.sqrt(2))))


# --------------------------------------------------------------------------
# Triggering on y
# --------------------------------------------------------------------------

def trigger_on_signal(tg, sg, Delta):
    """
    Send-on-delta applied to the sampled signal sg on grid tg.
    Segment-wise: between events the reference is fixed, so the next
    event is the first index where |sg - ref| >= Delta, found with a
    vectorised argmax on the remaining slice.
    """
    taus = []
    ref = sg[0]
    i = 0
    n = len(tg)
    while i < n - 1:
        rest = sg[i + 1:]
        hit = np.argmax(np.abs(rest - ref) >= Delta)
        if not (abs(rest[hit] - ref) >= Delta):
            break                                   # no further crossing
        k = i + 1 + hit
        target = ref + np.sign(sg[k] - ref) * Delta
        d = sg[k] - sg[k - 1]
        frac = 0.0 if d == 0 else np.clip((target - sg[k - 1]) / d, 0, 1)
        taus.append(tg[k - 1] + frac * (tg[k] - tg[k - 1]))
        ref = target
        i = k
    return np.asarray(taus)


def run_case(A, w, theta, Delta, sigma_e, w_c, n_periods=80,
             pts_per_period=6000, seed=0):
    theta = np.asarray(theta, float)
    C, psi = output_envelope(A, w, theta)
    T = n_periods * 2 * np.pi / w
    tg = np.linspace(0, T, int(n_periods * pts_per_period))

    x = C * np.sin(w * tg + psi)
    if sigma_e > 0:
        e, edot, sigma_de = make_noise(sigma_e, w_c,
                                       rng=np.random.default_rng(seed))
        y = x + e(tg)
    else:
        sigma_de = 0.0
        y = x

    tau = trigger_on_signal(tg, y, Delta)
    if len(tau) < 10:
        return None

    s = w * tau + psi
    zeta = np.mean(np.exp(2j * s))

    # Rice prediction for the mean rate: time-average of (*)
    xdot = C * w * np.cos(w * tg + psi)
    lam_rice = np.mean(folded_normal_mean(xdot, sigma_de)) / Delta

    return dict(tau=tau, N=len(tau), C=C, psi=psi, T=T,
                sigma_de=sigma_de, zeta=zeta,
                lam_meas=len(tau) / T, lam_rice=lam_rice)


# --------------------------------------------------------------------------
# Does least squares stay unbiased?
# --------------------------------------------------------------------------

def bias_check(A, w, theta, Delta, sigma_e, w_c, sigma_meas,
               shared_noise, n_runs=400, n_periods=40, seed0=100):
    """
    shared_noise=True  : the SAME realisation drives the trigger and the
                         measurement  -> selection bias expected
    shared_noise=False : trigger and measurement noise independent
                         -> no selection bias expected
    """
    theta = np.asarray(theta, float)
    est = []
    for r in range(n_runs):
        res = run_case(A, w, theta, Delta, sigma_e, w_c,
                       n_periods=n_periods, pts_per_period=2000, seed=seed0 + r)
        if res is None:
            continue
        tau = res["tau"]
        Phi = np.column_stack([A * np.sin(w * tau), A * w * np.cos(w * tau)])
        if shared_noise:
            e, _, _ = make_noise(sigma_e, w_c,
                                 rng=np.random.default_rng(seed0 + r))
            meas_noise = e(tau)              # same realisation as the trigger
        else:
            meas_noise = np.random.default_rng(10_000 + r).normal(
                0, sigma_meas, len(tau))
        y = Phi @ theta + meas_noise
        est.append(np.linalg.pinv(Phi.T @ Phi) @ Phi.T @ y)
    est = np.array(est)
    return est.mean(0), est.std(0) / np.sqrt(len(est)), len(est)


# --------------------------------------------------------------------------

def main():
    A, w = 1.0, 1.0
    theta = np.array([1.0, 0.5])
    C, psi = output_envelope(A, w, theta)
    Delta = C / 100
    w_c = 20.0 * w            # noise bandwidth

    print(f"C = {C:.4f},  Delta = {Delta:.5f},  noise bandwidth w_c = {w_c}\n")
    print("Sweeping noise level.  r = sigma_edot / (C*w)  is the ratio of")
    print("noise slope to signal slope -- the natural dimensionless group.\n")
    print(f"{'sigma_e/C':>10}{'sigma_de/(Cw)':>15}{'N':>8}"
          f"{'lam_meas':>11}{'lam_rice':>11}{'ratio':>8}"
          f"{'|zeta|':>9}{'arg z':>9}")

    rows = []
    for se_rel in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1):
        res = run_case(A, w, theta, Delta, se_rel * C, w_c)
        if res is None:
            continue
        r = res["sigma_de"] / (C * w)
        rows.append((r, abs(res["zeta"])))
        print(f"{se_rel:>10.4g}{r:>15.4g}{res['N']:>8d}"
              f"{res['lam_meas']:>11.2f}{res['lam_rice']:>11.2f}"
              f"{res['lam_meas']/res['lam_rice']:>8.3f}"
              f"{abs(res['zeta']):>9.4f}"
              f"{np.degrees(np.angle(res['zeta'])):>9.2f}")

    print("\n(|zeta| = 1/3 = 0.3333 is the noise-free value.)")

    print("\n" + "=" * 74)
    print("IS LEAST SQUARES STILL UNBIASED?   true theta = [1, 0.5]")
    print("=" * 74)
    for se_rel in (0.01, 0.05):
        for shared in (False, True):
            m, se, n = bias_check(A, w, theta, Delta, se_rel * C, w_c,
                                  sigma_meas=se_rel * C, shared_noise=shared,
                                  n_runs=300)
            tag = "SHARED (trigger=meas)" if shared else "independent"
            flag = ""
            if np.any(np.abs(m - theta) > 3 * se):
                flag = "  <-- significant bias"
            print(f"sigma_e/C={se_rel:<6} {tag:<24} "
                  f"mean={np.array2string(m, precision=4)} "
                  f"+-{np.array2string(3*se, precision=4)}{flag}")


if __name__ == "__main__":
    main()
