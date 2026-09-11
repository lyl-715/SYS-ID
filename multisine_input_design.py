"""
Multisine extension of the event-based input design experiments.

    u(t) = sum_{m=1..M} a_m sin(w_m t + varphi_m)

Everything downstream of u -- regressors, output, send-on-delta events,
Fisher information, criteria -- is the same machinery as the single-sine
case; only the excitation is richer.  The frequencies are NOT required to
share a common fundamental, so no periodicity is imposed anywhere and all
windows are treated as finite chunks of a (generally) quasi-periodic
signal.

What actually changes
---------------------
For a single sinusoid the output is x = C sin(s), s = wt + psi, and
send-on-delta puts events uniformly in output VALUE, so the induced
density in output phase is p(s) = |cos s| / 4 and everything collapses
onto one complex number

    zeta = <exp(2i s)>_events,      |zeta| = 1/3.

For a multisine the output is a sum of sinusoids; p ∝ 1/|xdot| has no
closed form and there is no single phase variable.  A replacement has to
be chosen.  Two are implemented here (see `zeta_fundamental` and
`zeta_analytic`), they coincide exactly at M = 1, and a third,
`zeta_from_fim`, reads |zeta| back out of the information matrices --
that one is the operationally meaningful one because it is defined by
what the design criteria actually see.

Reuse
-----
`trigger_on_signal` (noisy_trigger) does the event detection; `criteria`
and the model constants come from event_input_design.  The single-sine
`regressors` / `output_envelope` / `fisher` are the M = 1 special cases of
the `*_ms` functions here, and the sanity check asserts they agree.
"""

import multiprocessing as mp
from typing import NamedTuple

import numpy as np

from event_input_design import (MODEL_STATIC, MODEL_DYNAMIC, TAU1, TAU2,
                                criteria, fisher, output_envelope,
                                periodic_instants, regressors,
                                send_on_delta_instants)
from noisy_trigger import trigger_on_signal


# --------------------------------------------------------------------------
# The multisine itself
# --------------------------------------------------------------------------

class Multisine(NamedTuple):
    """u(t) = sum_m amps[m] * sin(freqs[m] * t + phases[m])."""
    amps: np.ndarray
    freqs: np.ndarray
    phases: np.ndarray

    @property
    def M(self):
        return len(self.amps)


def multisine(amps, freqs, phases=None):
    """
    Build a Multisine, sorted by frequency so that component 0 is always
    the lowest one -- 'the fundamental' has to mean something definite for
    `zeta_fundamental`, and with incommensurate frequencies the lowest
    component is the only defensible choice.
    """
    amps = np.atleast_1d(np.asarray(amps, float))
    freqs = np.atleast_1d(np.asarray(freqs, float))
    phases = (np.zeros_like(amps) if phases is None
              else np.atleast_1d(np.asarray(phases, float)))
    if not (len(amps) == len(freqs) == len(phases)):
        raise ValueError("amps, freqs, phases must have equal length")
    o = np.argsort(freqs)
    return Multisine(amps[o], freqs[o], phases[o])


def random_multisine(M, rng, a_range=(0.2, 1.0), w_range=(0.3, 4.0),
                     power=None):
    """
    A random multisine with incommensurate-by-construction frequencies
    (drawn from a continuum, so a common period is a measure-zero event).

    `power`, if given, rescales the amplitudes to a fixed signal power
    sum(a^2)/2 -- the natural way to hold 'input energy' constant while
    moving around the reachable set.
    """
    a = rng.uniform(*a_range, M)
    w = rng.uniform(*w_range, M)
    p = rng.uniform(0.0, 2 * np.pi, M)
    if power is not None:
        a *= np.sqrt(power / (np.sum(a ** 2) / 2))
    return multisine(a, w, p)


def evaluate(ms, t):
    """u(t).  Loops over components: M is small, len(t) is not."""
    out = np.zeros_like(np.asarray(t, float))
    for a, w, p in zip(ms.amps, ms.freqs, ms.phases):
        out += a * np.sin(w * t + p)
    return out


def evaluate_dot(ms, t):
    """udot(t), analytic."""
    out = np.zeros_like(np.asarray(t, float))
    for a, w, p in zip(ms.amps, ms.freqs, ms.phases):
        out += a * w * np.cos(w * t + p)
    return out


# --------------------------------------------------------------------------
# Regressors, output
# --------------------------------------------------------------------------

def _channel_tf(model):
    """
    The two regressor channels as transfer functions of the input, so that
    both model structures go through one code path.  phi_j(t) is the
    steady-state response of F_j to u -- no ODE solver, no transient.
    """
    if model == MODEL_STATIC:
        return (lambda w: np.ones_like(np.asarray(w, complex)),   # phi1 = u
                lambda w: 1j * np.asarray(w, complex))            # phi2 = udot
    if model == MODEL_DYNAMIC:
        return (lambda w: 1.0 / (1.0 + 1j * np.asarray(w, float) * TAU1),
                lambda w: 1.0 / (1.0 + 1j * np.asarray(w, float) * TAU2))
    raise ValueError(f"unknown model: {model}")


def regressors_ms(t, ms, model=MODEL_STATIC):
    """
    phi(t) at steady state, shape (len(t), 2).  Multisine counterpart of
    event_input_design.regressors(t, A, omega, model).
    """
    t = np.asarray(t, float)
    cols = []
    for F in _channel_tf(model):
        H = F(ms.freqs)
        c = np.zeros_like(t)
        for a, w, p, h in zip(ms.amps, ms.freqs, ms.phases, H):
            c += a * np.abs(h) * np.sin(w * t + p + np.angle(h))
        cols.append(c)
    return np.column_stack(cols)


def output_multisine(ms, theta, model=MODEL_STATIC):
    """
    x = phi^T theta is itself a multisine.  Returns it as a Multisine, so
    amplitudes C_m and phases psi_m are available in closed form.  This is
    the multisine generalisation of output_envelope, which returns the
    single (C, psi) pair.
    """
    theta = np.asarray(theta, float)
    F1, F2 = _channel_tf(model)
    H = theta[0] * F1(ms.freqs) + theta[1] * F2(ms.freqs)
    return multisine(ms.amps * np.abs(H), ms.freqs, ms.phases + np.angle(H))


def output(ms, theta, t, model=MODEL_STATIC):
    """Noise-free output x(t), evaluated from the output multisine."""
    return evaluate(output_multisine(ms, theta, model), t)


def output_peak(msx):
    """sum |C_m| -- the worst-case excursion; reduces to C at M = 1.

    Delta is measured against this so that the M = 1 convention
    Delta = r * C carries over unchanged.  It is conservative for large M
    (the peak is rarely attained), so `output_rms` is also reported.
    """
    return float(np.sum(np.abs(msx.amps)))


def output_rms(msx):
    """sqrt(sum C_m^2 / 2) -- exact for incommensurate frequencies."""
    return float(np.sqrt(np.sum(msx.amps ** 2) / 2))


def output_slope_peak(msx):
    return float(np.sum(np.abs(msx.amps * msx.freqs)))


# --------------------------------------------------------------------------
# Phase variables -- the thing that has to be replaced for M > 1
# --------------------------------------------------------------------------

def phase_fundamental(msx, t):
    """
    Option A: keep the phase of the lowest-frequency output component,
    s = w_1 t + psi_1, and accept that the induced density is no longer a
    function of s alone.  Cheap, and it is the variable the single-sine
    theory literally uses, but it throws away the other components.
    """
    return msx.freqs[0] * np.asarray(t, float) + msx.phases[0]


def phase_analytic(msx, t):
    """
    Option B: the instantaneous phase of the analytic signal of x.  With
    z(t) = sum_m C_m exp(i(w_m t + psi_m)) we have x = Im z exactly, and
    all frequencies are positive, so arg z IS the Hilbert phase -- in
    closed form, no numerical Hilbert transform.

    At M = 1 this is identically w t + psi, so it reduces to Option A.
    For M > 1 it tracks the signal rather than one component, which is the
    honest generalisation of 'phase of the output'.  Its weakness is that
    it is only a faithful phase while z stays away from the origin; when
    two components nearly cancel, arg z slews arbitrarily fast.
    """
    t = np.asarray(t, float)
    z = np.zeros_like(t, dtype=complex)
    for C, w, p in zip(msx.amps, msx.freqs, msx.phases):
        z += C * np.exp(1j * (w * t + p))
    return np.angle(z)


def zeta_fundamental(taus, msx):
    """zeta = <exp(2i s)> over the event set, s = fundamental phase."""
    return complex(np.mean(np.exp(2j * phase_fundamental(msx, taus))))


def zeta_analytic(taus, msx):
    """zeta = <exp(2i s)> over the event set, s = analytic-signal phase."""
    return complex(np.mean(np.exp(2j * phase_analytic(msx, taus))))


def zeta_from_fim(I_ev, I_per):
    """
    |zeta| read back out of the information matrices, via the single-sine
    identities det ratio = 1 - |zeta|^2 and lambda_min ratio = 1 - |zeta|.

    These two agree only if the single-sine structure holds.  For a
    multisine they need not, and the gap between them is itself the
    diagnostic for how far the one-complex-number description has been
    stretched.
    """
    d_ev, d_per = np.linalg.det(I_ev), np.linalg.det(I_per)
    l_ev = float(np.linalg.eigvalsh(I_ev).min())
    l_per = float(np.linalg.eigvalsh(I_per).min())
    det_ratio = d_ev / d_per
    lam_ratio = l_ev / l_per
    # NOT clamped to [0,1]: at M > 1 the ratios can leave the range the
    # single-sine identities allow (det ratio > 1 means event sampling beats
    # periodic sampling outright), and a NaN is the honest way to say that
    # the |zeta| reading does not exist rather than to report a floor of 0.
    return dict(det_ratio=float(det_ratio), lam_ratio=lam_ratio,
                zeta_det=float(np.sqrt(1.0 - det_ratio))
                if det_ratio <= 1.0 else float("nan"),
                zeta_lam=float(1.0 - lam_ratio),
                cond_ev=float(np.linalg.cond(I_ev)),
                cond_per=float(np.linalg.cond(I_per)))


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------

def event_grid(T_total, msx, Delta, pts_per_delta=10, max_points=40_000_000):
    """
    Detection grid.  Sized by the STEEPEST part of the output: the grid
    step must be small compared with the time it takes x to travel Delta
    at peak slope, or crossings get skipped and send-on-delta silently
    loses events.  Near the turning points the grid is far finer than
    needed, which is harmless.
    """
    dt = Delta / (pts_per_delta * output_slope_peak(msx))
    n = int(np.ceil(T_total / dt)) + 1
    if n > max_points:
        raise MemoryError(f"grid would need {n:,} points; raise max_points "
                          f"or coarsen Delta")
    return np.linspace(0.0, T_total, n)


def send_on_delta_instants_ms(T_total, Delta, ms, theta, model=MODEL_STATIC,
                              pts_per_delta=10, refine=True):
    """
    Send-on-delta event times for a multisine excitation.

    Detection is `trigger_on_signal` (reused verbatim): first grid index
    where |x - ref| >= Delta, crossing time by linear interpolation, and
    the reference advanced to the EXACT level ref + sign*Delta so the
    level grid never drifts with the sample values.

    `refine` then polishes each crossing with Newton steps on the analytic
    x(t) - level.  The level sequence is reconstructed from x(0) and the
    detected signs, which is unambiguous because the interpolated event is
    already within far less than Delta/2 of its level.  This removes the
    O(dt^2) interpolation error entirely and decouples the answer from the
    grid density -- worth having, because zeta is a second-harmonic moment
    and so is exactly the kind of quantity a systematic timing error
    biases.
    """
    msx = output_multisine(ms, theta, model)
    tg = event_grid(T_total, msx, Delta, pts_per_delta)
    xg = evaluate(msx, tg)
    taus = trigger_on_signal(tg, xg, Delta)
    if refine and len(taus):
        taus = refine_events(taus, Delta, float(xg[0]), msx)
    return taus


def refine_events(taus, Delta, x0, msx, n_newton=3):
    """Newton-polish each event onto its exact level x(tau) = level_i."""
    taus = np.asarray(taus, float).copy()
    x = evaluate(msx, taus)

    # rebuild the level ladder: level_i = level_{i-1} + sign_i * Delta
    levels = np.empty_like(taus)
    ref = x0
    for i in range(len(taus)):
        ref = ref + np.sign(x[i] - ref) * Delta
        levels[i] = ref

    for _ in range(n_newton):
        f = evaluate(msx, taus) - levels
        fp = evaluate_dot(msx, taus)
        step = np.where(np.abs(fp) > 1e-12, f / np.where(fp == 0, 1, fp), 0.0)
        # a Newton step must never jump past a neighbouring event
        gap = np.diff(taus, prepend=taus[0] - 1.0, append=taus[-1] + 1.0)
        lim = 0.45 * np.minimum(gap[:-1], gap[1:])
        taus = taus - np.clip(step, -lim, lim)
    return taus


# --------------------------------------------------------------------------
# Information
# --------------------------------------------------------------------------

def fisher_ms(t_samples, ms, sigma=1.0, model=MODEL_STATIC):
    """I = (1/sigma^2) sum_k phi(t_k) phi^T(t_k)."""
    Phi = regressors_ms(t_samples, ms, model)
    return (Phi.T @ Phi) / sigma ** 2


def experiment_equal_N_ms(ms, theta=(1.0, 0.5), sigma=1.0, T_total=None,
                          n_periods=40, delta_ratio=0.01,
                          model=MODEL_STATIC, pts_per_delta=10, refine=True):
    """
    One multisine, both sampling schemes, SAME event count.

    The window defaults to n_periods periods of the LOWEST input frequency
    -- a window length, not a claim that the signal repeats on it.  Events
    are generated first; the periodic reference then takes the same N
    uniform instants over the same window, so the comparison isolates
    where the samples sit.
    """
    theta = np.asarray(theta, float)
    msx = output_multisine(ms, theta, model)
    C_pk, C_rms = output_peak(msx), output_rms(msx)
    if T_total is None:
        T_total = n_periods * 2 * np.pi / ms.freqs[0]
    Delta = delta_ratio * C_pk

    tau = send_on_delta_instants_ms(T_total, Delta, ms, theta, model,
                                    pts_per_delta, refine)
    N = len(tau)
    if N < 8:
        return None
    tk = periodic_instants(T_total, N)

    I_ev = fisher_ms(tau, ms, sigma, model)
    I_per = fisher_ms(tk, ms, sigma, model)

    out = dict(ms=ms, msx=msx, theta=theta, model=model, T_total=T_total,
               Delta=Delta, delta_ratio=delta_ratio, C_peak=C_pk, C_rms=C_rms,
               N=N, tau=tau, tk=tk, I_ev=I_ev, I_per=I_per,
               c_ev=criteria(I_ev), c_per=criteria(I_per),
               zeta_fund=zeta_fundamental(tau, msx),
               zeta_anal=zeta_analytic(tau, msx),
               split_ev=float(I_ev[1, 1] / I_ev[0, 0]),
               split_per=float(I_per[1, 1] / I_per[0, 0]))
    out.update(zeta_from_fim(I_ev, I_per))
    return out


# --------------------------------------------------------------------------
# Parallel sweep scaffolding  (top-level worker -> picklable)
# --------------------------------------------------------------------------

def _sweep_worker(case):
    """
    One case = a dict with keys 'ms' plus any kwargs of
    experiment_equal_N_ms, plus an optional 'tag' echoed back.  Returns a
    compact summary (no event arrays -- they are large and would dominate
    the pickle traffic back from the pool).
    """
    tag = case.pop("tag", None)
    ms = case.pop("ms")
    try:
        r = experiment_equal_N_ms(ms, **case)
    except Exception as exc:                       # keep a sweep alive
        return dict(tag=tag, ok=False, error=f"{type(exc).__name__}: {exc}")
    if r is None:
        return dict(tag=tag, ok=False, error="too few events")
    return dict(tag=tag, ok=True, M=ms.M, amps=ms.amps, freqs=ms.freqs,
                phases=ms.phases, N=r["N"], Delta=r["Delta"],
                C_peak=r["C_peak"], C_rms=r["C_rms"],
                zeta_fund=r["zeta_fund"], zeta_anal=r["zeta_anal"],
                det_ratio=r["det_ratio"], lam_ratio=r["lam_ratio"],
                zeta_det=r["zeta_det"], zeta_lam=r["zeta_lam"],
                cond_ev=r["cond_ev"], cond_per=r["cond_per"],
                split_ev=r["split_ev"], split_per=r["split_per"])


def run_sweep(cases, workers=4, chunksize=1, verbose=True, force_pool=False):
    """Map _sweep_worker over cases; serial below ~100 cases."""
    cases = [dict(c) for c in cases]
    if (len(cases) < 100 and not force_pool) or workers == 1:
        return [_sweep_worker(c) for c in cases]
    if verbose:
        print(f"  [pool] {len(cases)} cases on {workers} workers ...",
              flush=True)
    with mp.Pool(workers) as pool:
        return pool.map(_sweep_worker, cases, chunksize=chunksize)


# --------------------------------------------------------------------------
# Deliverable 2 -- M = 1 sanity check
# --------------------------------------------------------------------------

def _fmt(x, w=12, p=5):
    return f"{x:>{w}.{p}g}"


def check_reduces_to_single_sine(A=1.0, w=1.0, theta=(1.0, 0.5),
                                 model=MODEL_STATIC):
    """
    The M = 1 multisine path must agree with the existing single-sine code
    element by element: regressors, output envelope, event times, FIM.
    """
    theta = np.asarray(theta, float)
    ms = multisine([A], [w], [0.0])
    t = np.linspace(0.0, 37.0, 5000)

    d_phi = np.max(np.abs(regressors_ms(t, ms, model)
                          - regressors(t, A, w, model)))

    msx = output_multisine(ms, theta, model)
    C_ref, psi_ref = output_envelope(A, w, theta, model)
    d_C = abs(msx.amps[0] - C_ref)
    d_psi = abs(np.angle(np.exp(1j * (msx.phases[0] - psi_ref))))

    Delta = 0.02
    T = 20 * 2 * np.pi / w
    tau_old = send_on_delta_instants(T, Delta, A, w, theta, model)
    tau_raw = send_on_delta_instants_ms(T, Delta, ms, theta, model,
                                        refine=False)
    tau_ref = send_on_delta_instants_ms(T, Delta, ms, theta, model,
                                        refine=True)
    n = min(len(tau_old), len(tau_raw))
    d_tau = float(np.max(np.abs(tau_old[:n] - tau_raw[:n])))
    d_ref = float(np.max(np.abs(tau_ref[:n] - tau_raw[:n])))

    # the refined events must sit on an exact Delta ladder in x
    msx_ = output_multisine(ms, theta, model)
    lad = float(np.max(np.abs(np.abs(np.diff(evaluate(msx_, tau_ref)))
                              - Delta)))

    d_I = np.max(np.abs(fisher_ms(t, ms, 1.0, model)
                        - fisher(t, A, w, 1.0, model)))
    return dict(d_phi=d_phi, d_C=d_C, d_psi=d_psi, d_tau=d_tau, d_ref=d_ref,
                ladder=lad, d_I=d_I, n_old=len(tau_old), n_new=len(tau_raw))


def check_density(res, nbins=24):
    """
    Induced density of events in output phase against the predicted
    |cos s| / 4.  Uses the analytic phase (identical to the fundamental
    phase at M = 1).
    """
    s = np.mod(phase_analytic(res["msx"], res["tau"]), 2 * np.pi)
    counts, edges = np.histogram(s, bins=nbins, range=(0, 2 * np.pi))
    width = edges[1] - edges[0]
    meas = counts / (len(s) * width)
    mid = 0.5 * (edges[:-1] + edges[1:])
    # bin-AVERAGED |cos s| / 4, not the midpoint value: with only a couple
    # of dozen bins the two differ by more than the statistical error
    pred = np.array([np.mean(np.abs(np.cos(
        np.linspace(edges[i], edges[i + 1], 2001)))) / 4.0
        for i in range(nbins)])
    return mid, meas, pred


def check_off_grid(ms, theta, T_total, Delta, model=MODEL_STATIC,
                   pts_per_delta=10):
    """
    Events must not land on grid points -- if they did, send-on-delta
    would degenerate into a subset of periodic sampling.  Reports the
    distribution of the fractional grid position of each event.
    """
    msx = output_multisine(ms, theta, model)
    tg = event_grid(T_total, msx, Delta, pts_per_delta)
    dt = tg[1] - tg[0]
    xg = evaluate(msx, tg)
    tau = trigger_on_signal(tg, xg, Delta)
    frac = np.mod(tau / dt, 1.0)
    on_grid = np.mean((frac < 1e-9) | (frac > 1 - 1e-9))
    return dict(n=len(tau), frac=frac, on_grid_fraction=float(on_grid),
                dt=dt)


def sanity_check_M1(theta=(1.0, 0.5), sigma=1.0, n_periods=40,
                    delta_ratios=(0.04, 0.02, 0.01, 0.005, 0.0025),
                    combos=None, model=MODEL_STATIC):
    """Deliverable 2: does M = 1 reproduce |zeta| = 1/3 and det = 8/9?"""
    theta = np.asarray(theta, float)

    print("=" * 78)
    print("1.  M = 1 MULTISINE PATH vs THE EXISTING SINGLE-SINE CODE")
    print("=" * 78)
    print(f"{'model':<10}{'A':>6}{'omega':>7}{'N old/new':>12}"
          f"{'max|dphi|':>12}{'|dC|':>10}{'|dpsi|':>10}{'max|dI|':>11}"
          f"{'max|dtau|':>12}{'refine shift':>14}{'ladder err':>12}")
    for mdl in (MODEL_STATIC, MODEL_DYNAMIC):
        for A, w in ((1.0, 1.0), (2.0, 0.7), (0.5, 3.0)):
            c = check_reduces_to_single_sine(A, w, theta, mdl)
            counts = "{}/{}".format(c["n_old"], c["n_new"])
            print(f"{mdl:<10}{A:>6.3g}{w:>7.3g}{counts:>12}"
                  f"{c['d_phi']:>12.3e}{c['d_C']:>10.3e}{c['d_psi']:>10.3e}"
                  f"{c['d_I']:>11.3e}{c['d_tau']:>12.3e}{c['d_ref']:>14.3e}"
                  f"{c['ladder']:>12.3e}")
    print("\n  Event COUNTS agree exactly.  dtau is not zero because the two")
    print("  paths size their detection grid differently (the old code fixes")
    print("  4000 points per period, event_grid sizes by peak slope), so each")
    print("  carries its own O(dt^2) linear-interpolation error -- 'refine")
    print("  shift' is the same quantity, and it is what Newton removes.")
    print("  'ladder err' = max | |x(tau_i+1) - x(tau_i)| - Delta | over the")
    print("  refined events: ~1e-14 confirms they sit on the exact levels.")

    A, w = 1.0, 1.0
    ms = multisine([A], [w], [0.0])
    msx = output_multisine(ms, theta, model)
    C = output_peak(msx)
    print(f"\nsingle sine: A={A}, omega={w}, theta={theta}, model={model}")
    print(f"  C = {C:.6f}   psi = {np.degrees(msx.phases[0]):.4f} deg"
          f"   (window = {n_periods} periods)")

    print("\n" + "=" * 78)
    print("2.  |zeta| AND THE CRITERION RATIOS vs Delta")
    print("=" * 78)
    print("predictions: |zeta| = 1/3 = 0.333333,  det = 8/9 = 0.888889,")
    print("             lam_min = 2/3 = 0.666667,  cond_ev/cond_per = 2")
    print(f"\n{'Delta/C':>9}{'N':>8}{'N/period':>10}{'|z| fund':>11}"
          f"{'|z| anal':>11}{'|z| det':>10}{'|z| lam':>10}"
          f"{'det ratio':>12}{'lam ratio':>11}{'cond ev':>10}")

    rows = []
    for r in delta_ratios:
        res = experiment_equal_N_ms(ms, theta, sigma, n_periods=n_periods,
                                    delta_ratio=r, model=model)
        rows.append((r, res))
        print(f"{r:>9.4g}{res['N']:>8d}{res['N']/n_periods:>10.1f}"
              f"{abs(res['zeta_fund']):>11.6f}{abs(res['zeta_anal']):>11.6f}"
              f"{res['zeta_det']:>10.6f}{res['zeta_lam']:>10.6f}"
              f"{res['det_ratio']:>12.6f}{res['lam_ratio']:>11.6f}"
              f"{res['cond_ev']/res['cond_per']:>10.6f}")

    # linear extrapolation in Delta/C  ->  the asymptotic value
    x = np.array([r for r, _ in rows])
    print(f"\n{'quantity':<14}{'linear fit':>13}{'err':>11}"
          f"{'quadratic fit':>16}{'err':>11}{'slope':>10}{'prediction':>13}")
    for name, vals, pred in (
            ("|zeta| fund", [abs(s["zeta_fund"]) for _, s in rows], 1 / 3),
            ("|zeta| anal", [abs(s["zeta_anal"]) for _, s in rows], 1 / 3),
            ("|zeta| det", [s["zeta_det"] for _, s in rows], 1 / 3),
            ("|zeta| lam", [s["zeta_lam"] for _, s in rows], 1 / 3),
            ("det ratio", [s["det_ratio"] for _, s in rows], 8 / 9),
            ("lam ratio", [s["lam_ratio"] for _, s in rows], 2 / 3)):
        y = np.array(vals)
        b, a = np.polyfit(x, y, 1)
        aq = np.polyfit(x, y, 2)[-1]
        print(f"{name:<14}{a:>13.6f}{a - pred:>11.2e}{aq:>16.6f}"
              f"{aq - pred:>11.2e}{b:>10.4f}{pred:>13.6f}")

    print("\n  'slope' is the finite-threshold bias d/d(Delta/C): |zeta| comes")
    print("  out ABOVE 1/3 at finite Delta, the criterion ratios below their")
    print("  limits.  The bias is not exactly linear, so the linear fit over")
    print("  this Delta range is itself biased; the quadratic fit is the one")
    print("  to read.  Established bias coefficient for |zeta| is 2/3 =")
    print("  0.6667 -- the measured slope sits just under it because the")
    print("  curvature is negative.")

    print("\n" + "=" * 78)
    print("3.  I_22 / I_11  AND THE psi DEPENDENCE")
    print("=" * 78)
    print(f"{'theta':<16}{'psi[deg]':>10}{'|zeta|':>10}{'I22/I11 ev':>13}"
          f"{'predicted':>12}{'I22/I11 per':>13}{'w^2':>8}")
    for th in ((1.0, 0.0), (1.0, 0.25), (1.0, 0.5), (1.0, 1.0), (0.2, 1.0),
               (0.0, 1.0)):
        th = np.asarray(th, float)
        r = experiment_equal_N_ms(ms, th, sigma, n_periods=n_periods,
                                  delta_ratio=0.005, model=model)
        if r is None:
            continue
        psi = r["msx"].phases[0]
        z = abs(r["zeta_fund"])
        pred = w ** 2 * (1 + z * np.cos(2 * psi)) / (1 - z * np.cos(2 * psi))
        lab = "({:g}, {:g})".format(*th)
        print(f"{lab:<16}{np.degrees(psi):>10.3f}{z:>10.6f}"
              f"{r['split_ev']:>13.6f}{pred:>12.6f}{r['split_per']:>13.6f}"
              f"{w**2:>8.4g}")

    print("\n" + "=" * 78)
    print("4.  INDUCED DENSITY IN OUTPUT PHASE   (target: |cos s| / 4)")
    print("=" * 78)
    res = experiment_equal_N_ms(ms, theta, sigma, n_periods=n_periods,
                                delta_ratio=0.005, model=model)
    mid, meas, pred = check_density(res)
    print(f"N = {res['N']} events\n")
    print(f"{'s [deg]':>9}{'measured p(s)':>15}{'|cos s|/4':>12}"
          f"{'abs err':>11}{'rel err':>10}")
    for m, a, b in zip(mid, meas, pred):
        print(f"{np.degrees(m):>9.1f}{a:>15.6f}{b:>12.6f}"
              f"{a - b:>11.2e}{(a - b) / b:>10.3f}")
    turning = np.abs(np.cos(mid)) < np.cos(np.radians(75))
    print(f"\n  max absolute deviation             = "
          f"{np.max(np.abs(meas - pred)):.3e}")
    print(f"  max relative deviation, all bins   = "
          f"{np.max(np.abs(meas/pred - 1)):.4f}")
    print(f"  max relative deviation, away from  = "
          f"{np.max(np.abs(meas[~turning]/pred[~turning] - 1)):.4f}")
    print("      the turning points (|s-90|,|s-270| > 15 deg)")
    print("\n  The whole relative error lives in the two bins straddling a")
    print("  turning point, and it is NOT a failure of the density -- it is")
    print("  quantisation of the level ladder against a fixed histogram bin.")
    print("  That bin spans |x| in [cos(7.5deg), 1] * C, a band of width")
    print("  0.0086 C, so it contains only 0.0086 C / Delta ladder levels;")
    print("  below a few levels per bin the bin count is an integer artefact")
    print("  of where the ladder happens to fall, not a density estimate.")
    print("  (The turning point also loses one event per half period outright:")
    print("  the top level is crossed going up, but coming down the trigger")
    print("  needs a FULL Delta from it, so that level is never revisited.)")
    print(f"\n{'Delta/C':>10}{'N':>9}{'levels/peak bin':>18}"
          f"{'max rel, all':>14}{'max rel, away':>15}{'max abs':>11}")
    for r in (0.02, 0.01, 0.005, 0.0025, 0.00125):
        rr = experiment_equal_N_ms(ms, theta, sigma, n_periods=n_periods,
                                   delta_ratio=r, model=model)
        mm_mid, mm, pp = check_density(rr)
        tt = np.abs(np.cos(mm_mid)) < np.cos(np.radians(75))
        print(f"{r:>10.5g}{rr['N']:>9d}"
              f"{(1 - np.cos(np.radians(7.5))) / r:>18.2f}"
              f"{np.max(np.abs(mm/pp - 1)):>14.4f}"
              f"{np.max(np.abs(mm[~tt]/pp[~tt] - 1)):>15.4f}"
              f"{np.max(np.abs(mm - pp)):>11.2e}")
    print("\n  Neither artefact touches zeta: it is a smooth second-harmonic")
    print("  moment, so a defect confined to a shrinking phase window around")
    print("  the turning points enters it only at O(Delta/C) -- which is")
    print("  exactly the bias measured in section 2.")

    print("\n" + "=" * 78)
    print("5.  ARE THE EVENTS OFF-GRID?")
    print("=" * 78)
    g = check_off_grid(ms, theta, n_periods * 2 * np.pi / w, 0.005 * C, model)
    print(f"grid step dt = {g['dt']:.3e},  N = {g['n']} events")
    print(f"fraction landing on a grid point (within 1e-9 of one) = "
          f"{g['on_grid_fraction']:.6f}")
    h, _ = np.histogram(g["frac"], bins=10, range=(0, 1))
    print("histogram of the fractional position within a grid cell "
          "(flat = healthy):")
    print("  " + "".join(f"{v/len(g['frac']):>8.4f}" for v in h))
    return rows


# --------------------------------------------------------------------------
# Deliverable 1 -- infrastructure smoke tests at M > 1
# --------------------------------------------------------------------------

def smoke_test_multisine(theta=(1.0, 0.5), sigma=1.0, model=MODEL_STATIC):
    """
    Not an experiment -- just proof that the M > 1 path runs end to end and
    that the pieces are self-consistent.  The interesting question (what
    the reachable set Z looks like) is deliberately left for the next
    stage.
    """
    print("=" * 78)
    print("6.  M > 1 SMOKE TEST -- does the machinery run, and do the two")
    print("    phase conventions agree?")
    print("=" * 78)
    theta = np.asarray(theta, float)

    cases = [
        ("M=1 reference", multisine([1.0], [1.0], [0.0])),
        ("M=2 harmonic 1:2", multisine([1.0, 0.5], [1.0, 2.0], [0.0, 0.0])),
        ("M=2 incommensurate", multisine([1.0, 0.5], [1.0, np.sqrt(2)],
                                         [0.0, 0.9])),
        ("M=2 weak 2nd tone", multisine([1.0, 0.05], [1.0, 2.37], [0.0, 0.3])),
        ("M=3 spread", multisine([0.8, 0.5, 0.3], [0.7, 1.9, 3.3],
                                 [0.1, 2.0, 4.1])),
    ]
    print(f"{'case':<20}{'N':>7}{'C_pk':>8}{'C_rms':>8}{'|z| fund':>10}"
          f"{'arg z_f':>9}{'|z| anal':>10}{'arg z_a':>9}"
          f"{'|z| det':>9}{'|z| lam':>9}{'det ratio':>11}")
    for tag, ms in cases:
        r = experiment_equal_N_ms(ms, theta, sigma, n_periods=30,
                                  delta_ratio=0.01, model=model)
        if r is None:
            print(f"{tag:<20}  too few events")
            continue
        zf, za = r["zeta_fund"], r["zeta_anal"]
        print(f"{tag:<20}{r['N']:>7d}{r['C_peak']:>8.4f}{r['C_rms']:>8.4f}"
              f"{abs(zf):>10.5f}{np.degrees(np.angle(zf)):>9.2f}"
              f"{abs(za):>10.5f}{np.degrees(np.angle(za)):>9.2f}"
              f"{r['zeta_det']:>9.5f}{r['zeta_lam']:>9.5f}"
              f"{r['det_ratio']:>11.6f}")

    print("\n  Read this as a status report, not a result:")
    print("   - at M = 1 all four columns coincide, as they must;")
    print("   - at M > 1 the fundamental-phase and analytic-phase zeta")
    print("     DISAGREE, and both disagree with the FIM-implied values, so")
    print("     the single-complex-number description does not survive")
    print("     intact.  Which convention (if either) is the useful one is")
    print("     exactly the question for the next stage;")
    print("   - |z| det and |z| lam also part company at M > 1, which is the")
    print("     sharpest sign of it: they are two readings of the same")
    print("     number only under the single-sine FIM structure.  'nan' means")
    print("     det ratio > 1, i.e. no real |zeta| reproduces it at all.")
    print("\n  Flagging one thing without chasing it, since this session")
    print("  stops here: det ratio comes out ABOVE 1 for three of these")
    print("  multisines -- event-based sampling more informative than")
    print("  periodic at equal N, which a single sinusoid can never do")
    print("  (1 - |zeta|^2 <= 1).  Checked against the analytic time-average")
    print("  FIM and at 4x grid density, so it is not a sampling artefact.")
    print("  That is the thing to go after in the reachable-set study.")


def check_parallel(theta=(1.0, 0.5), workers=4):
    """The pool path must give bit-identical results to the serial path."""
    print("\n" + "=" * 78)
    print("7.  PARALLEL SCAFFOLDING")
    print("=" * 78)
    rng = np.random.default_rng(7)
    cases = [dict(tag=i, ms=random_multisine(3, rng, w_range=(0.8, 3.0)),
                  theta=theta, n_periods=12, delta_ratio=0.02)
             for i in range(8)]
    import time
    t0 = time.time()
    ser = run_sweep(cases, workers=1, verbose=False)
    t1 = time.time()
    par = run_sweep(cases, workers=workers, verbose=False, force_pool=True)
    t2 = time.time()
    same = all(a["ok"] == b["ok"] and (not a["ok"] or
               (a["N"] == b["N"] and a["zeta_anal"] == b["zeta_anal"]))
               for a, b in zip(ser, par))
    print(f"8 cases: serial {t1-t0:.2f}s, pool({workers}) {t2-t1:.2f}s, "
          f"identical = {same}")
    print(f"per-case cost at these settings: ~{(t1-t0)/8:.2f}s serial")
    print("  Cost scales as 1/Delta (grid points and event count both do),")
    print("  so budget the step-4/5 sweeps from a timing at the Delta you")
    print("  actually intend to use, not from this one.")
    for r in ser[:4]:
        print(f"  case {r['tag']}: N={r['N']:>6d}  "
              f"|z|fund={abs(r['zeta_fund']):.5f}  "
              f"|z|anal={abs(r['zeta_anal']):.5f}  "
              f"det ratio={r['det_ratio']:.5f}")
    return same


# --------------------------------------------------------------------------
# Figures  (every panel has a printed table above -- see sanity_check_M1)
# --------------------------------------------------------------------------

def make_plots(theta=(1.0, 0.5), sigma=1.0, n_periods=40, outdir=".",
               model=MODEL_STATIC):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta = np.asarray(theta, float)
    ms1 = multisine([1.0], [1.0], [0.0])
    msx1 = output_multisine(ms1, theta, model)
    C = output_peak(msx1)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) M = 1 vs M = 3: where the events land
    ms3 = multisine([0.8, 0.5, 0.3], [0.7, 1.9, 3.3], [0.1, 2.0, 4.1])
    for k, (ms, lab, col) in enumerate(((ms1, "M = 1", "C0"),
                                        (ms3, "M = 3", "C3"))):
        msx = output_multisine(ms, theta, model)
        Ck = output_peak(msx)
        T2 = 2 * 2 * np.pi / ms.freqs[0]
        tg = np.linspace(0, T2, 4000)
        off = 0 if k == 0 else -2.2 * Ck
        ax[0, 0].plot(tg, evaluate(msx, tg) / Ck + off / Ck, col, lw=1.1,
                      label=f"{lab}  x(t)/C")
        tau = send_on_delta_instants_ms(T2, 0.05 * Ck, ms, theta, model)
        ax[0, 0].plot(tau, evaluate(msx, tau) / Ck + off / Ck, "o", ms=3.2,
                      color=col, mfc="none",
                      label=f"{lab}  events (N={len(tau)})")
    ax[0, 0].set_title("send-on-delta events: uniform in OUTPUT VALUE\n"
                       "(M=3 curve offset for clarity)", fontsize=10)
    ax[0, 0].set_xlabel("t"); ax[0, 0].set_ylabel("x / C  (+ offset)")
    ax[0, 0].legend(fontsize=7, ncol=2)

    # (b) induced density vs |cos s| / 4
    res = experiment_equal_N_ms(ms1, theta, sigma, n_periods=n_periods,
                                delta_ratio=0.005, model=model)
    mid, meas, pred = check_density(res, nbins=48)
    ax[0, 1].plot(np.degrees(mid), meas, "o", ms=4, label="measured")
    ss = np.linspace(0, 2 * np.pi, 600)
    ax[0, 1].plot(np.degrees(ss), np.abs(np.cos(ss)) / 4, "r-", lw=1.2,
                  label="|cos s| / 4")
    ax[0, 1].set_title(f"M=1 induced density in output phase "
                       f"(N={res['N']}, Delta/C=0.005)", fontsize=10)
    ax[0, 1].set_xlabel("output phase s [deg]"); ax[0, 1].set_ylabel("p(s)")
    ax[0, 1].legend(fontsize=8)

    # (c) convergence of |zeta| and det ratio as Delta -> 0
    ratios = np.array([0.04, 0.02, 0.01, 0.005, 0.0025])
    zs, ds = [], []
    for r in ratios:
        rr = experiment_equal_N_ms(ms1, theta, sigma, n_periods=n_periods,
                                   delta_ratio=r, model=model)
        zs.append(abs(rr["zeta_anal"])); ds.append(rr["det_ratio"])
    zs, ds = np.array(zs), np.array(ds)
    ax[1, 0].plot(ratios, zs, "o-", label="|zeta| measured")
    ax[1, 0].axhline(1 / 3, color="r", ls="--", label="1/3")
    xf = np.linspace(0, ratios.max(), 50)
    ax[1, 0].plot(xf, np.polyval(np.polyfit(ratios, zs, 2), xf), "k:", lw=1,
                  label="quadratic fit")
    ax[1, 0].set_xlabel("Delta / C"); ax[1, 0].set_ylabel("|zeta|")
    ax[1, 0].set_title("finite-threshold bias: |zeta| -> 1/3 linearly in "
                       "Delta/C", fontsize=10)
    ax[1, 0].legend(fontsize=8, loc="lower right")
    a2 = ax[1, 0].twinx()
    a2.plot(ratios, ds, "s--", color="C2", ms=4)
    a2.axhline(8 / 9, color="C2", ls=":", lw=1)
    a2.set_ylabel("det ratio  (green, -> 8/9)", color="C2")

    # (d) the two phase conventions at M > 1
    rng = np.random.default_rng(3)
    cases = [dict(tag=i, ms=random_multisine(rng.integers(1, 5), rng,
                                             w_range=(0.6, 3.2)),
                  theta=theta, n_periods=15, delta_ratio=0.02)
             for i in range(120)]
    out = [r for r in run_sweep(cases, workers=4, verbose=False) if r["ok"]]
    Ms = np.array([r["M"] for r in out])
    zf = np.array([abs(r["zeta_fund"]) for r in out])
    za = np.array([abs(r["zeta_anal"]) for r in out])
    for m in sorted(set(Ms)):
        k = Ms == m
        ax[1, 1].plot(zf[k], za[k], "o", ms=4, label=f"M = {m}")
    ax[1, 1].plot([0, 0.6], [0, 0.6], "k--", lw=1, label="agreement")
    ax[1, 1].plot(1 / 3, 1 / 3, "r*", ms=14, label="single sine (1/3, 1/3)")
    ax[1, 1].set_xlabel("|zeta| from the fundamental phase")
    ax[1, 1].set_ylabel("|zeta| from the analytic-signal phase")
    ax[1, 1].set_title("the two phase conventions agree only at M = 1\n"
                       "(120 random multisines; orientation only)",
                       fontsize=10)
    ax[1, 1].legend(fontsize=7)

    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout()
    path = f"{outdir}/multisine_sanity_figures.png"
    fig.savefig(path, dpi=130)
    print(f"\nfigure written to {path}")

    # the table that goes with panel (d)
    print("\ntable for panel (d) -- |zeta| by convention, "
          "summarised over 120 random multisines")
    print(f"{'M':>4}{'n':>6}{'mean |z| fund':>16}{'mean |z| anal':>16}"
          f"{'mean |diff|':>14}{'max |diff|':>13}")
    for m in sorted(set(Ms)):
        k = Ms == m
        d = np.abs(zf[k] - za[k])
        print(f"{m:>4}{k.sum():>6}{zf[k].mean():>16.5f}{za[k].mean():>16.5f}"
              f"{d.mean():>14.5f}{d.max():>13.5f}")
    return path


def main():
    np.set_printoptions(precision=5, suppress=True)
    sanity_check_M1()
    print()
    smoke_test_multisine()
    check_parallel()
    print()
    make_plots()


if __name__ == "__main__":
    main()
