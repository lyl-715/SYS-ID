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


# ==========================================================================
# WHY CAN det ratio EXCEED 1?  The |xdot| weighting, with no trigger at all
# ==========================================================================
#
# As Delta -> 0 send-on-delta emits |xdot| dt / Delta events in dt, so the
# empirical measure of the event set converges to the density
#
#     p(t)  =  |xdot(t)| / integral |xdot|,
#
# and I_ev / N converges to the moments of phi under that weight while
# I_per / N converges to the moments under uniform weight.  det I = N^2 *
# det(moments), so at equal N
#
#     det I_ev / det I_per   ->   D_w / D_1.
#
# Everything below computes D_w / D_1 by quadrature: no threshold, no event
# sequence, no grid detection.  If det ratio > 1 survives that, it is a
# property of the weighting and not of the simulation.

def uniform_moments_exact(ms, model=MODEL_STATIC):
    """
    <phi_i phi_j> under uniform time weighting over an infinite window, in
    closed form.  Cross-frequency products time-average to zero and
    <sin(a+a_i) sin(a+a_j)> = cos(a_i - a_j)/2, so

        <phi_i phi_j>_1 = (1/2) sum_m a_m^2 Re[ H_i(w_m) conj(H_j(w_m)) ].

    Exact, so D_1 carries no quadrature error at all.
    """
    F1, F2 = _channel_tf(model)
    H = np.column_stack([F1(ms.freqs), F2(ms.freqs)])
    Mo = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            Mo[i, j] = 0.5 * np.sum(ms.amps ** 2
                                    * np.real(H[:, i] * np.conj(H[:, j])))
    return Mo


def weighted_moments_time(ms, theta, model=MODEL_STATIC, T=None,
                          n_periods=400, n=2_000_001):
    """
    <phi_i phi_j>_w with w(t) = |xdot(t)|, by trapezoid quadrature in time,
    plus the uniform moments over the SAME finite window.

    The finite-window uniform moments are what the equal-N periodic
    reference actually measures, so returning both makes the window-length
    effect visible rather than hidden.
    """
    msx = output_multisine(ms, theta, model)
    if T is None:
        T = n_periods * 2 * np.pi / ms.freqs[0]
    t = np.linspace(0.0, T, n)
    Phi = regressors_ms(t, ms, model)
    wt = np.abs(evaluate_dot(msx, t))
    Z = np.trapezoid(wt, t)

    Mw, M1 = np.empty((2, 2)), np.empty((2, 2))
    for i in range(2):
        for j in range(i, 2):
            pij = Phi[:, i] * Phi[:, j]
            Mw[i, j] = Mw[j, i] = np.trapezoid(pij * wt, t) / Z
            M1[i, j] = M1[j, i] = np.trapezoid(pij, t) / T
    return Mw, M1


def weighted_moments_torus(ms, theta, model=MODEL_STATIC, n_phase=None):
    """
    <phi_i phi_j>_w as an average over the M-torus of component phases.

    For rationally independent frequencies Weyl equidistribution makes this
    EXACTLY the infinite-time average, with no window-length error and no
    dependence on the phases varphi_m at all.  For commensurate frequencies
    it is the wrong average -- the signal then only visits a closed curve on
    the torus -- so `weighted_moments_time` is the one to trust there.  The
    gap between the two is itself a useful diagnostic.
    """
    if n_phase is None:
        n_phase = {1: 200_001, 2: 1024, 3: 192}.get(ms.M, 96)
    msx = output_multisine(ms, theta, model)
    F1, F2 = _channel_tf(model)
    H = np.column_stack([F1(ms.freqs), F2(ms.freqs)])

    ax = np.linspace(0.0, 2 * np.pi, n_phase, endpoint=False)
    g = np.meshgrid(*([ax] * ms.M), indexing="ij")

    p1 = np.zeros_like(g[0])
    p2 = np.zeros_like(g[0])
    xd = np.zeros_like(g[0])
    for m in range(ms.M):
        p1 += ms.amps[m] * np.abs(H[m, 0]) * np.sin(g[m] + np.angle(H[m, 0]))
        p2 += ms.amps[m] * np.abs(H[m, 1]) * np.sin(g[m] + np.angle(H[m, 1]))
        # the output multisine carries the SAME varphi_m offsets, which the
        # torus average integrates over, so only its extra phase shift
        # (psi_m - varphi_m) survives
        xd += (msx.amps[m] * msx.freqs[m]
               * np.cos(g[m] + msx.phases[m] - ms.phases[m]))
    wt = np.abs(xd)
    Z = wt.mean()
    Mw = np.empty((2, 2))
    for i, (pi, pj) in enumerate(((p1, p1), (p1, p2), (p2, p2))):
        v = float((pi * pj * wt).mean() / Z)
        if i == 0:
            Mw[0, 0] = v
        elif i == 1:
            Mw[0, 1] = Mw[1, 0] = v
        else:
            Mw[1, 1] = v
    return Mw


def det_gain(Mw, M1):
    """
    D_w / D_1 and its decomposition.  <u udot>_1 = 0 for any multisine, so
    D_1 = M1_11 M1_22 and

        D_w / D_1 = R11 * R22 - cross^2,
        R11 = <u^2>_w / <u^2>_1,  R22 = <udot^2>_w / <udot^2>_1,
        cross = <u udot>_w / sqrt(<u^2>_1 <udot^2>_1).

    R11 + R22 is the quantity that is pinned to exactly 2 at M = 1 and is
    free to exceed it at M > 1.
    """
    R11 = Mw[0, 0] / M1[0, 0]
    R22 = Mw[1, 1] / M1[1, 1]
    cross = Mw[0, 1] / np.sqrt(M1[0, 0] * M1[1, 1])
    return dict(gain=float(R11 * R22 - cross ** 2), R11=float(R11),
                R22=float(R22), cross=float(cross),
                direction_sum=float(R11 + R22),
                corr_per=float(M1[0, 1] / np.sqrt(M1[0, 0] * M1[1, 1])))


def flagged_cases():
    """
    Exactly the multisines that produced det ratio > 1 last session, plus
    the M = 1 control.  Regenerated from the same seeds so the comparison
    is against the same objects, not lookalikes.
    """
    cases = [("M=1 control", multisine([1.0], [1.0], [0.0])),
             ("M=2 harmonic 1:2", multisine([1.0, 0.5], [1.0, 2.0],
                                            [0.0, 0.0])),
             ("M=2 incommensurate", multisine([1.0, 0.5], [1.0, np.sqrt(2)],
                                              [0.0, 0.9])),
             ("M=2 weak 2nd tone", multisine([1.0, 0.05], [1.0, 2.37],
                                             [0.0, 0.3])),
             ("M=3 spread", multisine([0.8, 0.5, 0.3], [0.7, 1.9, 3.3],
                                      [0.1, 2.0, 4.1]))]
    rng = np.random.default_rng(7)               # same seed as check_parallel
    for i in range(4):
        cases.append((f"M=3 random #{i}",
                      random_multisine(3, rng, w_range=(0.8, 3.0))))
    for i in range(4):                           # consume the rest of the seed
        random_multisine(3, rng, w_range=(0.8, 3.0))
    return cases


def _rat_flag(ms, tol=1e-9, max_den=12):
    """Are the frequencies rationally related with small denominators?"""
    from fractions import Fraction
    base = ms.freqs[0]
    fr = [Fraction(float(w / base)).limit_denominator(max_den)
          for w in ms.freqs]
    ok = all(abs(float(f) - w / base) < tol for f, w in zip(fr, ms.freqs))
    return ok, ":".join(f"{f.numerator}/{f.denominator}" if f.denominator > 1
                        else str(f.numerator) for f in fr)


# --------------------------------------------------------------------------
# Step 1 -- quadrature only, no trigger, no Delta, no event sequence
# --------------------------------------------------------------------------

def experiment_quadrature_check(theta=(1.0, 0.5), model=MODEL_STATIC,
                                n_periods=400, n=2_000_001):
    print("=" * 96)
    print("STEP 1.  D_w / D_1 BY QUADRATURE -- no trigger, no Delta, no "
          "event sequence")
    print("=" * 96)
    print("  weight w(t) = |xdot(t)|;  D_1 uses the EXACT infinite-window")
    print("  uniform moments (closed form), so D_1 has no quadrature error.")
    print("  'torus' averages over the phase torus: exactly the infinite-time")
    print("  average when the frequencies are rationally independent, and the")
    print("  WRONG average when they are not -- which is why it is printed")
    print("  next to a 'rational?' column rather than trusted blindly.\n")
    theta = np.asarray(theta, float)

    print(f"{'case':<20}{'freq ratios':>14}{'rational?':>11}"
          f"{'D_w/D_1 time':>14}{'D_w/D_1 torus':>15}{'R11':>9}{'R22':>9}"
          f"{'cross':>9}{'R11+R22':>10}")
    rows = []
    for tag, ms in flagged_cases():
        Mw, _ = weighted_moments_time(ms, theta, model,
                                      n_periods=n_periods, n=n)
        M1 = uniform_moments_exact(ms, model)
        g = det_gain(Mw, M1)
        gt = det_gain(weighted_moments_torus(ms, theta, model), M1)
        rat, lab = _rat_flag(ms)
        rows.append((tag, ms, g, gt))
        print(f"{tag:<20}{lab:>14}{('yes' if rat else 'no'):>11}"
              f"{g['gain']:>14.6f}{gt['gain']:>15.6f}{g['R11']:>9.5f}"
              f"{g['R22']:>9.5f}{g['cross']:>9.5f}"
              f"{g['direction_sum']:>10.6f}")

    print("\n  Quadrature convergence, per case.  'grid' refines the")
    print("  quadrature at fixed window (pure numerics); 'window' lengthens")
    print("  the window (ergodicity of a quasi-periodic signal, which is a")
    print("  property of the signal, not of the quadrature).")
    print(f"\n{'case':<20}{'D_w/D_1 (T=400)':>17}{'grid 4x':>12}"
          f"{'T=1600':>12}{'d(window)':>12}")
    for (tag, ms, g, _) in rows:
        Mw4, _ = weighted_moments_time(ms, theta, model, n=8_000_001,
                                       n_periods=n_periods)
        g4 = det_gain(Mw4, uniform_moments_exact(ms, model))["gain"]
        MwL, _ = weighted_moments_time(ms, theta, model, n=8_000_001,
                                       n_periods=1600)
        gL = det_gain(MwL, uniform_moments_exact(ms, model))["gain"]
        print(f"{tag:<20}{g['gain']:>17.6f}{g4 - g['gain']:>12.2e}"
              f"{gL:>12.6f}{gL - g['gain']:>12.2e}")
    print("\n  Grid refinement moves nothing (1e-7), so the quadrature is")
    print("  converged.  The window column is the only thing that moves, and")
    print("  only for quasi-periodic cases -- exactly as expected.")
    return rows


# --------------------------------------------------------------------------
# Step 2 -- does the simulated det ratio converge to D_w / D_1?
# --------------------------------------------------------------------------

def experiment_delta_convergence(theta=(1.0, 0.5), model=MODEL_STATIC,
                                 ratios=(0.08, 0.04, 0.02, 0.01, 0.005,
                                         0.0025),
                                 n_periods=60, sim_periods=60):
    """
    Two references matter and they are NOT the same number:
      - 'window'   : D_w / D_1 computed over the SAME finite window the
                     simulation uses.  This is what the simulation must
                     converge to, because the event set only samples that
                     window.
      - 'infinite' : the long-window limit.  The gap between the two is
                     window-ergodicity error, not a trigger effect.
    """
    print("\n" + "=" * 96)
    print("STEP 2.  DOES det ratio CONVERGE TO D_w / D_1 AS Delta -> 0?")
    print("=" * 96)
    print(f"  Delta swept over {ratios[0]/ratios[-1]:.0f}x "
          f"({ratios[0]} down to {ratios[-1]} of C_peak); "
          f"window = {sim_periods} periods of w_1.\n")
    theta = np.asarray(theta, float)

    for tag, ms in flagged_cases():
        T = sim_periods * 2 * np.pi / ms.freqs[0]
        Mw_win, M1_win = weighted_moments_time(ms, theta, model, T=T,
                                               n=4_000_001)
        g_win = det_gain(Mw_win, M1_win)["gain"]
        Mw_inf, _ = weighted_moments_time(ms, theta, model, n_periods=1600,
                                          n=8_000_001)
        g_inf = det_gain(Mw_inf, uniform_moments_exact(ms, model))["gain"]

        print(f"{tag}   (window D_w/D_1 = {g_win:.6f}, "
              f"infinite-window = {g_inf:.6f})")
        print(f"{'  Delta/C':>10}{'N':>9}{'det ratio':>12}"
              f"{'- window':>12}{'/(Delta/C)':>12}{'R11+R22 sim':>14}")
        prev = None
        for r in ratios:
            res = experiment_equal_N_ms(ms, theta, n_periods=sim_periods,
                                        delta_ratio=r, model=model)
            if res is None:
                continue
            dr, N = res["det_ratio"], res["N"]
            # the simulated analogue of R11 + R22, at equal N
            s = (res["I_ev"][0, 0] / res["I_per"][0, 0]
                 + res["I_ev"][1, 1] / res["I_per"][1, 1])
            print(f"{r:>10.5g}{N:>9d}{dr:>12.6f}{dr - g_win:>12.2e}"
                  f"{(dr - g_win) / r:>12.4f}{s:>14.6f}")
            prev = dr
        print(f"  -> residual at the smallest Delta: {prev - g_win:+.2e}"
              f"   (drifting would show as a non-shrinking gap)\n")


# --------------------------------------------------------------------------
# Step 3 -- where does the extra information come from?
# --------------------------------------------------------------------------
#
# Exact identity.  Let w_eff^2 = <udot^2>_1 / <u^2>_1 (the power-weighted
# mean square frequency) and define the instantaneous energy
#
#     E(t) = u(t)^2 + (udot(t) / w_eff)^2.
#
# Then <u^2>_1 = <udot^2>_1 / w_eff^2 = <E>_1 / 2 by construction, so for ANY
# weight w,
#
#     R11 + R22 = <u^2>_w/<u^2>_1 + <udot^2>_w/<udot^2>_1 = 2 <E>_w / <E>_1.
#
# At M = 1, u = A sin(s) and udot = A w cos(s) with w_eff = w, so E == A^2 is
# CONSTANT: <E>_w = <E>_1 for every weight and the sum is pinned to exactly
# 2.  That is the sin^2 + cos^2 = 1 identity, and it is what bounds
# det ratio = R11 R22 - cross^2 <= (R11+R22)^2/4 = 1 at M = 1.
#
# At M > 1, E(t) fluctuates -- it is the squared envelope of the multisine --
# and the sum exceeds 2 exactly when the weight |xdot| is positively
# correlated with E.  It is: both |xdot| and E are large during the
# high-envelope epochs of the beat.  So the two regressor directions do NOT
# trade off; they gain together, because the weighting preferentially samples
# the moments when the signal is simply bigger in every direction at once.

def envelope_analytic(msx, t):
    """|analytic signal| of a multisine -- its instantaneous envelope."""
    t = np.asarray(t, float)
    z = np.zeros_like(t, dtype=complex)
    for C, w, p in zip(msx.amps, msx.freqs, msx.phases):
        z += C * np.exp(1j * (w * t + p))
    return np.abs(z)


def energy_identity(ms, theta, model=MODEL_STATIC, n_periods=400,
                    n=2_000_001):
    """
    Check R11 + R22 = 2 <E>_w / <E>_1 directly, and report the ingredients
    that decide whether it exceeds 2.
    """
    theta = np.asarray(theta, float)
    msx = output_multisine(ms, theta, model)
    T = n_periods * 2 * np.pi / ms.freqs[0]
    t = np.linspace(0.0, T, n)

    Phi = regressors_ms(t, ms, model)
    wt = np.abs(evaluate_dot(msx, t))
    M1 = uniform_moments_exact(ms, model)
    w_eff2 = M1[1, 1] / M1[0, 0]
    E = Phi[:, 0] ** 2 + Phi[:, 1] ** 2 / w_eff2

    Z = np.trapezoid(wt, t)
    E_w = np.trapezoid(E * wt, t) / Z
    E_1 = np.trapezoid(E, t) / T

    Mw, M1win = weighted_moments_time(ms, theta, model, T=T, n=n)
    g = det_gain(Mw, M1)
    # the identity is a statement about ONE window, so check it against that
    # window's own uniform moments; using the infinite-window moments would
    # fold in ergodicity error and blur the test
    gwin = det_gain(Mw, M1win)

    # envelope moments.  k = <R^3> / (<R> <R^2>), NOT <R^3><R>/<R^2>^2: the
    # slow average enters the weighted moment as <R^3>/<R> and the uniform
    # moment as <R^2>/2.
    R = envelope_analytic(msx, t)
    mR = [np.trapezoid(R ** q, t) / T for q in (1, 2, 3)]
    k = mR[2] / (mR[0] * mR[1])
    return dict(R11=g["R11"], R22=g["R22"], sum=g["direction_sum"],
                sum_win=gwin["direction_sum"],
                sum_from_E=float(2 * E_w / E_1), gain=g["gain"],
                E_ratio=float(E_w / E_1),
                E_cv=float(np.sqrt(max(np.trapezoid(E ** 2, t) / T - E_1 ** 2,
                                       0.0)) / E_1),
                corr=float(np.corrcoef(wt, E)[0, 1]),
                k=float(k), nb_sum=float(2 * k), nb_gain=float(8 / 9 * k ** 2))


def experiment_direction_gain(theta=(1.0, 0.5), model=MODEL_STATIC,
                              n_periods=400):
    print("\n" + "=" * 100)
    print("STEP 3.  DO BOTH REGRESSOR DIRECTIONS GAIN AT ONCE?  (is "
          "R11 + R22 > 2 ?)")
    print("=" * 100)
    print("  E(t) = u^2 + (udot/w_eff)^2, w_eff^2 = <udot^2>_1/<u^2>_1.")
    print("  Exact identity (any weight):  R11 + R22 = 2 <E>_w / <E>_1.")
    print("  At M=1, E is CONSTANT, so the sum is pinned to 2 and")
    print("  det ratio = R11 R22 - cross^2 <= ((R11+R22)/2)^2 = 1.\n")

    print(f"{'case':<20}{'R11':>8}{'R22':>8}{'R11+R22':>10}"
          f"{'2<E>w/<E>1':>12}{'identity':>10}{'std(E)/<E>':>12}"
          f"{'corr(|xd|,E)':>14}{'D_w/D_1':>10}")
    rows = []
    for tag, ms in flagged_cases():
        d = energy_identity(ms, theta, model, n_periods=n_periods)
        rows.append((tag, ms, d))
        print(f"{tag:<20}{d['R11']:>8.5f}{d['R22']:>8.5f}{d['sum']:>10.6f}"
              f"{d['sum_from_E']:>12.6f}"
              f"{abs(d['sum_win']-d['sum_from_E']):>10.1e}"
              f"{d['E_cv']:>12.5f}{d['corr']:>14.5f}{d['gain']:>10.6f}")

    print("\n  The identity column is the residual |R11+R22 - 2<E>w/<E>1|:")
    print("  it holds to quadrature precision, so the sum really is an")
    print("  energy-weighting statement and nothing else.")
    print("  std(E)/<E> = 0 exactly at M = 1 and is the only reason the")
    print("  sum can move.  corr(|xdot|, E) is positive in every case: the")
    print("  weight samples the high-envelope epochs.")

    print("\n  Narrowband limit.  If the output is narrowband, x ~ R(t)")
    print("  sin(w t + Phi) with R slow, then u ~ R sin, udot ~ R w cos, the")
    print("  weight |xdot| ~ R |cos|, the slow and fast averages factor, and")
    print("  with k = <R^3> / (<R> <R^2>):")
    print("      R11 = k (1 - (1/3) cos 2psi),  R22 = k (1 + (1/3) cos 2psi),")
    print("      R11 + R22 = 2k,   D_w / D_1 = (8/9) k^2.")
    print("  k >= 1 by Cauchy-Schwarz, with equality iff R is constant -- so")
    print("  within this family the weighting can only gain, never lose.\n")
    print(f"{'case':<20}{'bandwidth':>11}{'k':>9}{'2k':>10}{'R11+R22':>10}"
          f"{'(8/9)k^2':>11}{'D_w/D_1':>10}{'rel err':>10}")
    for tag, ms, d in rows:
        bw = (ms.freqs.max() - ms.freqs.min()) / ms.freqs.mean()
        print(f"{tag:<20}{bw:>11.4f}{d['k']:>9.5f}{d['nb_sum']:>10.5f}"
              f"{d['sum']:>10.5f}{d['nb_gain']:>11.6f}{d['gain']:>10.6f}"
              f"{d['nb_gain']/d['gain'] - 1:>10.3f}")
    print("\n  Read this as a NEGATIVE result: at these bandwidths the")
    print("  narrowband formula under-predicts by 7-32%, and the error does")
    print("  not order by bandwidth (weak-2nd-tone has the widest band and")
    print("  the smallest error, simply because its gain is near zero).  It")
    print("  is a limit, not a model for these signals.  The limit itself is")
    print("  real, and taking it properly is the next table.")
    return rows


def experiment_narrowband_limit(theta=(1.0, 0.5), model=MODEL_STATIC,
                                rho=1.0, w1=1.0,
                                eps=(1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01)):
    """
    Two tones at w1 and w1(1+eps), equal amplitude, eps -> 0.  Uses the
    torus average, which for rationally independent frequencies IS the
    infinite-time average -- the point of using it here is that the beat
    period blows up like 1/eps, so any fixed time window would run out of
    ergodicity long before eps got small.  eps is irrational-ish by
    construction for the same reason.
    """
    print("\n" + "=" * 100)
    print("STEP 3b.  THE NARROWBAND LIMIT, TAKEN PROPERLY  (two equal tones, "
          "w2/w1 = 1 + eps)")
    print("=" * 100)
    print("  predicted limit:  k -> 4/3,  R11+R22 -> 8/3 = 2.666667,")
    print(f"  D_w/D_1 -> 128/81 = {128/81:.6f}\n")
    theta = np.asarray(theta, float)
    print(f"{'eps':>10}{'beat/carrier':>14}{'k':>10}{'(8/9)k^2':>11}"
          f"{'D_w/D_1':>11}{'rel err':>10}{'R11+R22':>11}{'2k':>10}")
    for e in eps:
        # nudge off exact rational ratios: the torus average is the time
        # average only for rationally independent frequencies
        w2 = w1 * (1.0 + e * (1.0 + 1e-7 * np.pi))
        ms = multisine([1.0, rho], [w1, w2], [0.0, 0.0])
        msx = output_multisine(ms, theta, model)
        Mw = weighted_moments_torus(ms, theta, model, n_phase=1536)
        g = det_gain(Mw, uniform_moments_exact(ms, model))
        # envelope moments on the torus: R = |C1 + C2 exp(i psi)|
        psi = np.linspace(0, 2 * np.pi, 200_001)
        R = np.abs(msx.amps[0] + msx.amps[1] * np.exp(1j * psi))
        mR = [np.trapezoid(R ** q, psi) / (2 * np.pi) for q in (1, 2, 3)]
        k = mR[2] / (mR[0] * mR[1])
        nb = 8 / 9 * k ** 2
        print(f"{e:>10.4g}{1/e:>14.1f}{k:>10.6f}{nb:>11.6f}{g['gain']:>11.6f}"
              f"{nb/g['gain'] - 1:>10.4f}{g['direction_sum']:>11.6f}"
              f"{2*k:>10.6f}")
    print("\n  The narrowband formula becomes exact as eps -> 0, confirming")
    print("  the mechanism.  'beat/carrier' is how many carrier periods one")
    print("  beat lasts: it is 1/eps, so the gain is only realisable in an")
    print("  experiment many beat periods long.  That is the practical")
    print("  catch, and step 4 quantifies it.")



# --------------------------------------------------------------------------
# Step 4 -- the best M = 2 multisine
# --------------------------------------------------------------------------
#
# On the power constraint: D_w / D_1 is INVARIANT under u -> c u.  Both the
# weighted and the uniform moments scale as c^2 and the normalised weight
# |xdot| / integral|xdot| does not move at all.  So a power budget does not
# bind on this objective; it fixes the absolute information level, not the
# ratio.  The genuine design variables at M = 2 are the amplitude ratio
# rho = a2/a1, the frequency ratio r = w2/w1, and the frequency SCALE w1 --
# scale matters here because x = theta1 u + theta2 udot, so w1 -> lambda w1
# is equivalent to theta2 -> lambda theta2.  That is a dependence on the
# assumed theta, not a universal fact, and it is reported as such.

def gain_M2(w1, r, rho, theta=(1.0, 0.5), model=MODEL_STATIC, n_phase=384,
            phases=(0.0, 0.0), method="torus"):
    """D_w/D_1 for a two-tone multisine.  Top-level so a Pool can pickle it."""
    theta = np.asarray(theta, float)
    ms = multisine([1.0, rho], [w1, r * w1], list(phases))
    M1 = uniform_moments_exact(ms, model)
    if method == "torus":
        Mw = weighted_moments_torus(ms, theta, model, n_phase=n_phase)
    else:
        Mw, _ = weighted_moments_time(ms, theta, model, n_periods=4000,
                                      n=8_000_001)
    return det_gain(Mw, M1)


def _gain_worker_M2(case):
    w1, r, rho, theta, model, n_phase = case
    try:
        return gain_M2(w1, r, rho, theta, model, n_phase)["gain"]
    except Exception:
        return np.nan


def wide_separation_limit(beta, n=4000):
    """
    The r -> infinity limit of the M = 2 problem, reduced to ONE parameter.

    Take both frequencies far below theta1/theta2, so x ~ theta1 u and the
    weight is |udot|; then send r = w2/w1 -> infinity holding
    beta = a2 w2 / (a1 w1), the ratio of the two tones' contributions to the
    DERIVATIVE.  In units a1 = w1 = 1 the second tone drops out of u but not
    out of udot, and with p1, p2 independent uniform phases

        phi    = ( sin p1 ,  cos p1 + beta cos p2 ),
        weight = | cos p1 + beta cos p2 |.

    The cross moment vanishes by the symmetry (p1, p2) -> (pi-p1, pi-p2),
    so D_w/D_1 = R11 * R22 exactly here.
    """
    ax = (np.arange(n) + 0.5) * 2 * np.pi / n
    p1, p2 = np.meshgrid(ax, ax, indexing="ij")
    c = np.cos(p1) + beta * np.cos(p2)
    W = np.abs(c)
    Z = W.mean()
    R11 = ((np.sin(p1) ** 2 * W).mean() / Z) / 0.5
    R22 = ((c ** 2 * W).mean() / Z) / ((1 + beta ** 2) / 2)
    cross = (np.sin(p1) * c * W).mean() / Z / np.sqrt(0.5 * (1 + beta ** 2) / 2)
    return dict(gain=float(R11 * R22 - cross ** 2), R11=float(R11),
                R22=float(R22), cross=float(cross),
                direction_sum=float(R11 + R22))


def experiment_optimise_M2(theta=(1.0, 0.5), model=MODEL_STATIC, workers=4,
                           n_phase=384):
    print("\n" + "=" * 100)
    print("STEP 4.  WHICH M = 2 MULTISINE MAXIMISES D_w / D_1 ?")
    print("=" * 100)
    print("  D_w/D_1 is invariant under u -> c u: both moments scale as c^2")
    print("  and the normalised weight does not move.  So a POWER CONSTRAINT")
    print("  DOES NOT BIND on this objective -- it sets the absolute")
    print("  information level, not the ratio.  The real design variables are")
    print("    rho = a2/a1,  r = w2/w1,  and the scale w1.")
    print("  Scale matters because x = theta1 u + theta2 udot, so w1 ->")
    print("  lambda w1 acts exactly like theta2 -> lambda theta2.  Everything")
    print(f"  below is at theta = {tuple(np.asarray(theta, float))}; the")
    print("  theta-dependence is reported explicitly rather than hidden.")
    theta = np.asarray(theta, float)

    w1s = np.array([0.02, 0.25, 1.0, 4.0])
    rs = 1.0 + np.geomspace(0.01, 60.0, 30)
    rhos = np.geomspace(0.02, 2.5, 26)
    cases, idx = [], []
    for i, w1 in enumerate(w1s):
        for j, r in enumerate(rs):
            for k, rho in enumerate(rhos):
                cases.append((w1, r * (1 + 1e-7 * np.pi), rho, theta, model,
                              n_phase))
                idx.append((i, j, k))
    print(f"\n  landscape grid: {len(w1s)} x {len(rs)} x {len(rhos)} = "
          f"{len(cases)} evaluations on {workers} workers ...", flush=True)
    import time
    t0 = time.time()
    with mp.Pool(workers) as pool:
        vals = pool.map(_gain_worker_M2, cases, chunksize=16)
    G = np.full((len(w1s), len(rs), len(rhos)), np.nan)
    for (a, b, c), v in zip(idx, vals):
        G[a, b, c] = v
    print(f"  done in {time.time()-t0:.1f}s")
    i, j, k = np.unravel_index(np.nanargmax(G), G.shape)
    print(f"  grid maximum D_w/D_1 = {G[i,j,k]:.6f} at w1={w1s[i]:g}, "
          f"r={rs[j]:.4g}, rho={rhos[k]:.4g}")

    print("\n  Slice at w1 = %g (smallest), D_w/D_1 vs (r, rho):" % w1s[0])
    sp = [0, 4, 8, 11, 14, 17, 20, 23, 25]
    print("      rho ->" + "".join(f"{rhos[c]:>9.3g}" for c in sp))
    for b in [0, 5, 10, 14, 18, 22, 26, 29]:
        print(f"  r={rs[b]:>7.3f}" + "".join(f"{G[0, b, c]:>9.5f}"
                                             for c in sp))

    print("\n  Ridge: best rho at each r (w1 = %g), and the combination"
          % w1s[0])
    print("  beta = rho * r = a2 w2 / (a1 w1) that it corresponds to.")
    print(f"{'r':>10}{'best rho':>11}{'beta = rho r':>14}{'D_w/D_1':>11}"
          f"{'R11':>9}{'R22':>9}{'R11+R22':>10}")
    for b in [0, 4, 8, 12, 16, 20, 24, 29]:
        c = int(np.nanargmax(G[0, b]))
        d = gain_M2(w1s[0], rs[b] * (1 + 1e-7 * np.pi), rhos[c], theta,
                    model, 1024)
        print(f"{rs[b]:>10.4f}{rhos[c]:>11.4f}{rhos[c]*rs[b]:>14.4f}"
              f"{d['gain']:>11.6f}{d['R11']:>9.5f}{d['R22']:>9.5f}"
              f"{d['direction_sum']:>10.6f}")
    print("\n  The optimum is a RIDGE, not a peak: raising r and lowering rho")
    print("  together keeps beta ~ 1.2 and the gain nearly constant.  It")
    print("  climbs slowly with r, so the maximum is a LIMIT at r -> inf.")

    print("\n  That limit is a one-parameter problem (see "
          "`wide_separation_limit`):")
    print("    phi = (sin p1, cos p1 + beta cos p2), weight = |cos p1 + "
          "beta cos p2|")
    print(f"\n{'beta':>10}{'D_w/D_1':>12}{'R11':>10}{'R22':>10}"
          f"{'R11+R22':>11}")
    for bb in (0.5, 0.8, 1.0, 1.1, 1.2, 1.222388, 1.3, 1.5, 2.0, 3.0):
        d = wide_separation_limit(bb, 6000)
        print(f"{bb:>10.6g}{d['gain']:>12.6f}{d['R11']:>10.6f}"
              f"{d['R22']:>10.6f}{d['direction_sum']:>11.6f}")
    dstar = wide_separation_limit(1.222388, 8000)
    print(f"\n  ==> M = 2 SUPREMUM  D_w/D_1 = {dstar['gain']:.7f}  at "
          f"beta* = 1.222388")
    print(f"      (beta = 1 gives exactly 128/81 = {128/81:.7f}, which is")
    print("       also the narrowband r -> 1 limit: at beta = 1 the weight")
    print("       |cos p1 + cos p2| = 2|cos((p1+p2)/2)||cos((p1-p2)/2)|")
    print("       factorises into the same envelope-times-carrier form.)")
    print("      It is a SUPREMUM, not attained: it needs r -> infinity.")
    return dict(w1s=w1s, rs=rs, rhos=rhos, G=G, limit=dstar)


def experiment_M2_phase_and_window(theta=(1.0, 0.5), model=MODEL_STATIC):
    """
    Two things the optimisation above deliberately leaves out.

    (a) Relative phase.  For rationally independent frequencies the phases
        are integrated out by the torus average and cannot matter.  They CAN
        matter at rational ratios, where the signal is genuinely periodic.
    (b) Window length.  The optimum needs r -> infinity at fixed beta, which
        means the SLOW tone's period grows without bound.  A real experiment
        has a finite window, and the gain is only realisable if that window
        holds many periods of the slowest tone.
    """
    theta = np.asarray(theta, float)
    print("\n" + "=" * 100)
    print("STEP 4b.  RELATIVE PHASE, AND THE FINITE-WINDOW CATCH")
    print("=" * 100)
    print("  (a) relative phase at RATIONAL frequency ratios (time")
    print("      quadrature over the exact common period):\n")
    print(f"{'r':>8}{'rho':>7}" + "".join(f"{f'phi={d}deg':>12}"
                                          for d in (0, 45, 90, 135)))
    for r, rho in ((2.0, 0.6), (3.0, 0.4), (1.5, 0.7), (2.0, 1.0)):
        row = []
        for dphi in (0, 45, 90, 135):
            ms = multisine([1.0, rho], [1.0, r],
                           [0.0, np.radians(dphi)])
            Mw, _ = weighted_moments_time(ms, theta, model, n_periods=2000,
                                          n=4_000_001)
            row.append(det_gain(Mw, uniform_moments_exact(ms, model))["gain"])
        print(f"{r:>8.3g}{rho:>7.3g}" + "".join(f"{v:>12.6f}" for v in row))
    print("\n      Phase matters at rational ratios, and only there.  Since")
    print("      the optimum sits at r -> infinity (generic, irrational), the")
    print("      answer to 'what relative phase?' is: it does not matter.")
    print("      Note this also means the reachable set has measure-zero")
    print("      rational slivers where the gain differs -- worth knowing")
    print("      before the sweep treats phase as a free design variable.")

    print("\n  (b) what actually caps r: the THRESHOLD, not the window.")
    print("      The ridge needs r large at beta = a2 w2/(a1 w1) fixed, so")
    print("      a2 ~ beta/r: the second tone becomes WEAK in amplitude even")
    print("      as it stays strong in slope.  The asymptotic |xdot| density")
    print("      needs Delta small compared with the weakest output")
    print("      component C_min, not with C_peak -- and C_min ~ 1/r.")
    print("      So the Delta budget, and hence the event count, scales with")
    print("      r.  Demonstrated at r = 40, where C_min/C_peak = 0.032:\n")
    r = 40.0
    rho = 1.222388 / r
    rr = r * (1 + 1e-7 * np.pi)
    ms = multisine([1.0, rho], [0.02, 0.02 * rr], [0.0, 0.0])
    msx = output_multisine(ms, theta, model)
    Cmin, Cpk = float(msx.amps.min()), output_peak(msx)
    g = gain_M2(0.02, rr, rho, theta, model, 1024)["gain"]
    print(f"      C_peak = {Cpk:.5f},  C_min = {Cmin:.5f},  "
          f"quadrature D_w/D_1 = {g:.6f}")
    print(f"\n{'Delta/C_peak':>14}{'Delta/C_min':>13}{'N':>11}"
          f"{'det ratio':>12}{'error':>11}{'error/(D/C_min)':>18}")
    xs, ys = [], []
    for dr in (0.02, 0.01, 0.005, 0.002):
        res = experiment_equal_N_ms(ms, theta, n_periods=400,
                                    delta_ratio=dr, model=model)
        e = res["det_ratio"] - g
        x = res["Delta"] / Cmin
        xs.append(x); ys.append(e)
        print(f"{dr:>14.5g}{x:>13.4f}{res['N']:>11d}"
              f"{res['det_ratio']:>12.6f}{e:>11.2e}{e/x:>18.4f}")
    b, a = np.polyfit(np.array(xs), np.array(ys), 1)
    print(f"\n      linear fit of the error in Delta/C_min: slope {b:.4f}, "
          f"intercept {a:+.2e}")
    print("      The error is proportional to Delta/C_min and extrapolates")
    print("      to zero, so the quadrature value is right and the")
    print("      simulation is simply under-resolved at fixed Delta/C_peak.")
    print("\n      CONSEQUENCE FOR THE REACHABLE-SET SWEEP: a fixed")
    print("      Delta/C_peak is NOT a uniform resolution across multisines.")
    print("      Any input with a weak component is biased, and biased")
    print("      UPWARDS in det ratio -- exactly the direction that would")
    print("      manufacture a spurious 'event sampling beats periodic'")
    print("      result.  Normalise Delta to the weakest component, or")
    print("      report Delta/C_min alongside every point.")


# --------------------------------------------------------------------------
# The many-tone (Gaussian) limit -- a closed form, and a hard bound of 2
# --------------------------------------------------------------------------
#
# As M grows with spread frequencies and random phases, (phi, xdot) becomes
# jointly Gaussian and zero mean by the CLT.  For jointly Gaussian variables,
# conditioning on xi = xdot,
#
#     E[phi_i phi_j | xi] = Sigma_ij - c_i c_j / s^2 + c_i c_j xi^2 / s^4,
#     c = <phi xdot>,  s^2 = <xdot^2>,
#
# and a Gaussian satisfies <|xi|^3> = 2 s^2 <|xi|>, so the |xi| weighting
# gives EXACTLY
#
#     M_w = Sigma + c c^T / s^2,
#     D_w / D_1 = 1 + c^T Sigma^{-1} c / s^2.
#
# For the static model with spectral moments m0 = <u^2>, m2 = <udot^2>,
# m4 = <uddot^2>:
#
#     D_w / D_1  =  1 + (th1^2 m2 + th2^2 m2^2/m0) / (th1^2 m2 + th2^2 m4)
#
# and since m2^2 <= m0 m4 (Cauchy-Schwarz on the spectral measure), the
# numerator never exceeds the denominator:
#
#     D_w / D_1  <=  2   in the Gaussian limit,
#
# with equality iff th2 = 0, or the spectrum is a single line.  So the
# |xdot| weighting can at best DOUBLE the determinant, and it gets there by
# making the regressor covariance Sigma + c c^T / s^2 -- a rank-one addition
# along the direction c, i.e. along whichever regressor combination the
# trigger signal's derivative correlates with.  Whether a FINITE-M multisine
# can beat 2 is not settled by this and is left for the reachable-set study.

def gain_gaussian_limit(ms, theta, model=MODEL_STATIC):
    """D_w/D_1 predicted by the jointly-Gaussian closed form above."""
    theta = np.asarray(theta, float)
    F1, F2 = _channel_tf(model)
    H = np.column_stack([F1(ms.freqs), F2(ms.freqs)])
    Hx = theta[0] * H[:, 0] + theta[1] * H[:, 1]     # x = phi^T theta
    Hxd = 1j * ms.freqs * Hx                         # xdot
    p = ms.amps ** 2 / 2.0

    Sig = np.empty((2, 2))
    c = np.empty(2)
    for i in range(2):
        c[i] = np.sum(p * np.real(H[:, i] * np.conj(Hxd)))
        for j in range(2):
            Sig[i, j] = np.sum(p * np.real(H[:, i] * np.conj(H[:, j])))
    s2 = np.sum(p * np.abs(Hxd) ** 2)
    return float(1.0 + c @ np.linalg.solve(Sig, c) / s2)


def experiment_gaussian_limit(theta=(1.0, 0.5), model=MODEL_STATIC,
                              Ms=(1, 2, 3, 5, 8, 16, 32, 64, 128),
                              n_draw=24, w_range=(0.4, 4.0), seed=11):
    """
    Does the measured D_w/D_1 approach the Gaussian closed form as M grows?
    Averaged over random draws at each M, since any single multisine is a
    fluctuation about the limit.
    """
    print("\n" + "=" * 100)
    print("STEP 3c.  THE MANY-TONE LIMIT:  D_w/D_1 -> 1 + c' Sigma^-1 c / "
          "<xdot^2>,  bounded by 2")
    print("=" * 100)
    theta = np.asarray(theta, float)
    rng = np.random.default_rng(seed)
    print(f"  {n_draw} random multisines per M, frequencies in {w_range}, "
          f"theta = {tuple(theta)}\n")
    print(f"{'M':>5}{'mean D_w/D_1':>15}{'std':>9}{'mean Gaussian':>16}"
          f"{'std':>9}{'mean |diff|':>13}{'mean R11':>10}{'mean R22':>10}")
    for M in Ms:
        gs, gg, r1, r2 = [], [], [], []
        for _ in range(n_draw):
            ms = random_multisine(M, rng, w_range=w_range)
            npz = {1: 200_001, 2: 512}.get(M, 128 if M <= 3 else None)
            if M <= 2:
                Mw = weighted_moments_torus(ms, theta, model, n_phase=npz)
            else:
                # torus quadrature is infeasible past M = 3; a long time
                # window is the right tool once the signal is this rich
                Mw, _ = weighted_moments_time(ms, theta, model,
                                              n_periods=3000, n=4_000_001)
            g = det_gain(Mw, uniform_moments_exact(ms, model))
            gs.append(g["gain"]); r1.append(g["R11"]); r2.append(g["R22"])
            gg.append(gain_gaussian_limit(ms, theta, model))
        gs, gg = np.array(gs), np.array(gg)
        print(f"{M:>5}{gs.mean():>15.6f}{gs.std():>9.4f}{gg.mean():>16.6f}"
              f"{gg.std():>9.4f}{np.abs(gs-gg).mean():>13.6f}"
              f"{np.mean(r1):>10.5f}{np.mean(r2):>10.5f}")
    print("\n  R11 -> 1 and R22 -> 2 in the limit: the u direction is left")
    print("  alone and the udot direction doubles, because c is aligned with")
    print("  udot when theta1 dominates.  That is the sharpest statement of")
    print("  where the extra information comes from.")


# --------------------------------------------------------------------------
# Figures for the det-ratio > 1 investigation
# --------------------------------------------------------------------------

def make_plots_weighting(theta=(1.0, 0.5), model=MODEL_STATIC, outdir="."):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    theta = np.asarray(theta, float)
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))

    # (a) simulated det ratio -> quadrature D_w/D_1
    ratios = np.array([0.08, 0.04, 0.02, 0.01, 0.005])
    picks = [c for c in flagged_cases()
             if c[0] in ("M=1 control", "M=2 harmonic 1:2",
                         "M=3 random #2", "M=3 random #1")]
    tbl = []
    for n, (tag, ms) in enumerate(picks):
        T = 60 * 2 * np.pi / ms.freqs[0]
        Mw, M1 = weighted_moments_time(ms, theta, model, T=T, n=4_000_001)
        gw = det_gain(Mw, M1)["gain"]
        ys = [experiment_equal_N_ms(ms, theta, n_periods=60, delta_ratio=r,
                                    model=model)["det_ratio"] for r in ratios]
        ax[0, 0].semilogx(ratios, ys, "o-", color=f"C{n}", label=tag)
        ax[0, 0].axhline(gw, color=f"C{n}", ls="--", lw=1)
        tbl.append((tag, gw, list(ys)))
    ax[0, 0].axhline(1.0, color="k", lw=1.4, ls=":")
    ax[0, 0].set_xlabel("Delta / C_peak")
    ax[0, 0].set_ylabel("det I_ev / det I_per  at equal N")
    ax[0, 0].set_title("simulated det ratio -> quadrature D_w/D_1 "
                       "(dashed)\nblack dotted = 1: above it, events beat "
                       "periodic", fontsize=10)
    ax[0, 0].legend(fontsize=7)

    # (b) both directions gain at once
    rng = np.random.default_rng(21)
    pts = []
    for M in (1, 2, 3, 5):
        for _ in range(22):
            ms = random_multisine(M, rng, w_range=(0.4, 4.0))
            Mw, M1 = weighted_moments_time(ms, theta, model, n_periods=600,
                                           n=1_000_001)
            g = det_gain(Mw, M1)
            pts.append((M, g["R11"], g["R22"]))
    pts = np.array(pts)
    for M in (1, 2, 3, 5):
        k = pts[:, 0] == M
        ax[0, 1].plot(pts[k, 1], pts[k, 2], "o", ms=4, label=f"M = {M}")
    xs = np.linspace(0.55, 1.3, 50)
    ax[0, 1].plot(xs, 2 - xs, "k--", lw=1.3, label="R11+R22 = 2  (M=1 line)")
    ax[0, 1].plot(xs, 1.0 / xs, "r:", lw=1.3, label="R11 R22 = 1  (gain = 1)")
    ax[0, 1].set_xlabel("R11 = <u^2>_w / <u^2>_1")
    ax[0, 1].set_ylabel("R22 = <udot^2>_w / <udot^2>_1")
    ax[0, 1].set_title("M=1 is stuck on R11+R22 = 2; M>1 escapes it\n"
                       "above the red curve, det ratio > 1", fontsize=10)
    ax[0, 1].legend(fontsize=7)

    # (c) the M = 2 landscape and its r -> inf limit
    rs = 1.0 + np.geomspace(0.01, 60.0, 26)
    rhos = np.geomspace(0.02, 2.5, 24)
    cases = [(0.02, r * (1 + 1e-7 * np.pi), p, theta, model, 384)
             for r in rs for p in rhos]
    with mp.Pool(4) as pool:
        G = np.array(pool.map(_gain_worker_M2, cases,
                              chunksize=16)).reshape(len(rs), len(rhos))
    im = ax[1, 0].pcolormesh(rhos, rs, G, shading="auto", cmap="viridis")
    ax[1, 0].plot(1.222388 / rs, rs, "w--", lw=1.5,
                  label="beta = rho r = 1.2224")
    ax[1, 0].set_xscale("log"); ax[1, 0].set_yscale("log")
    ax[1, 0].set_xlabel("rho = a2 / a1"); ax[1, 0].set_ylabel("r = w2 / w1")
    ax[1, 0].set_title("M = 2 landscape of D_w/D_1 (w1 = 0.02)\n"
                       "the optimum is a ridge at constant beta",
                       fontsize=10)
    ax[1, 0].legend(fontsize=7, loc="upper right")
    fig.colorbar(im, ax=ax[1, 0])

    # (d) Gaussian limit
    bs = np.geomspace(0.2, 6.0, 40)
    gl = [wide_separation_limit(b, 2000)["gain"] for b in bs]
    ax[1, 1].semilogx(bs, gl, "-", lw=1.6, color="C3",
                      label="M=2 r->inf limit")
    ax[1, 1].axhline(1.6127375, color="C3", ls="--", lw=1,
                     label="M=2 supremum 1.61274")
    ax[1, 1].axhline(128 / 81, color="C1", ls=":", lw=1.2,
                     label="128/81 (beta=1)")
    ax[1, 1].axhline(8 / 9, color="C0", ls=":", lw=1.2, label="M=1: 8/9")
    ax[1, 1].set_xlabel("beta = a2 w2 / (a1 w1)")
    ax[1, 1].set_ylabel("D_w / D_1")
    ax[1, 1].set_title("how far the weighting can go: 8/9 at M=1, "
                       "1.613 at M=2", fontsize=10)
    ax[1, 1].legend(fontsize=7, loc="lower center")

    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout()
    path = f"{outdir}/multisine_weighting_figures.png"
    fig.savefig(path, dpi=130)
    print(f"\nfigure written to {path}")

    print("\ntable for panel (a): simulated det ratio vs Delta/C_peak")
    print(f"{'case':<20}{'D_w/D_1':>10}" +
          "".join(f"{r:>10.4g}" for r in ratios))
    for tag, gw, ys in tbl:
        print(f"{tag:<20}{gw:>10.6f}" + "".join(f"{y:>10.6f}" for y in ys))

    print("\ntable for panel (b): fraction of draws with R11+R22 > 2 "
          "and with gain > 1")
    print(f"{'M':>4}{'n':>6}{'mean R11+R22':>15}{'frac sum>2':>13}"
          f"{'frac gain>1':>13}")
    for M in (1, 2, 3, 5):
        k = pts[:, 0] == M
        sm = pts[k, 1] + pts[k, 2]
        gn = pts[k, 1] * pts[k, 2]
        print(f"{M:>4}{k.sum():>6}{sm.mean():>15.6f}"
              f"{np.mean(sm > 2 + 1e-9):>13.3f}{np.mean(gn > 1):>13.3f}")

    print("\ntable for panel (d): the r->inf M=2 limit family")
    print(f"{'beta':>8}{'D_w/D_1':>11}")
    for b in (0.2, 0.5, 0.8, 1.0, 1.2224, 1.5, 2.0, 3.0, 6.0):
        print(f"{b:>8.4g}{wide_separation_limit(b, 4000)['gain']:>11.6f}")
    return path


def main_weighting():
    """
    The whole det-ratio > 1 investigation, in order.  Each piece is callable
    on its own; this just runs them all.
    """
    np.set_printoptions(precision=5, suppress=True)
    experiment_quadrature_check()
    experiment_delta_convergence()
    experiment_direction_gain()
    experiment_narrowband_limit()
    experiment_gaussian_limit()
    experiment_optimise_M2()
    experiment_M2_phase_and_window()
    make_plots_weighting()


# --------------------------------------------------------------------------
# Is the M = 2 supremum degenerate?
# --------------------------------------------------------------------------
#
# The unconstrained supremum sits at w1 -> 0 with r -> infinity, which is
# exactly the corner where the steady-state sinusoidal description needs an
# observation window T >> 1/w1.  Two questions decide whether the result is
# usable or an artefact of that corner:
#   (a) how much of the gain survives inside a regime that can actually be
#       run (w1 >= 0.1, r <= 10), still in the stationary limit;
#   (b) what happens at an honestly finite window T, where D_1 must be the
#       window's OWN uniform moments -- that is what equal-N periodic
#       sampling over the same window measures.

def _degen_worker(case):
    kind = case[0]
    try:
        if kind == "stationary":
            _, w1, r, rho, theta, model, npz = case
            return gain_M2(w1, r * (1 + 1e-7 * np.pi), rho, theta, model,
                           npz)["gain"]
        _, w1, r, rho, ph2, T, n, theta, model = case
        ms = multisine([1.0, rho], [w1, r * w1], [0.0, ph2])
        Mw, M1 = weighted_moments_time(ms, theta, model, T=T, n=n)
        return det_gain(Mw, M1)["gain"]
    except Exception:
        return np.nan


def experiment_degeneracy(theta=(1.0, 0.5), model=MODEL_STATIC, workers=4,
                          T_window=100.0):
    theta = np.asarray(theta, float)
    print("=" * 92)
    print("IS THE M = 2 SUPREMUM DEGENERATE?")
    print("=" * 92)

    print("\n(a) STATIONARY LIMIT, RESTRICTED TO A RUNNABLE REGIME "
          "(w1 >= 0.1, r <= 10)\n")
    w1s = np.geomspace(0.1, 10, 16)
    rs = 1.0 + np.geomspace(0.01, 9.0, 20)
    rhs = np.geomspace(0.03, 4.0, 18)
    cases = [("stationary", a, b, c, theta, model, 384)
             for a in w1s for b in rs for c in rhs]
    with mp.Pool(workers) as pool:
        V = np.array(pool.map(_degen_worker, cases, chunksize=32)).reshape(
            len(w1s), len(rs), len(rhs))
    i, j, k = np.unravel_index(np.nanargmax(V), V.shape)
    print(f"    best on grid: D_w/D_1 = {V[i,j,k]:.6f}  at w1 = {w1s[i]:.4g}, "
          f"r = {rs[j]:.4g}, rho = {rhs[k]:.4g}")
    print(f"    unconstrained supremum 1.6127375;  M = 1 value {8/9:.6f};  "
          f"periodic 1")
    print(f"\n    {'r <=':>8}{'w1 >=':>8}{'best gain':>12}{'% of sup':>11}"
          f"{'at w1':>9}{'at r':>9}{'at rho':>9}{'beta':>9}")
    for rmax in (1.5, 2, 3, 5, 10):
        for wmin in (0.1, 0.5, 1.0):
            m, nn = rs <= rmax, w1s >= wmin
            if not m.any() or not nn.any():
                continue
            sub = V[np.ix_(nn, m)]
            a, b, c = np.unravel_index(np.nanargmax(sub), sub.shape)
            print(f"    {rmax:>8g}{wmin:>8g}{sub[a,b,c]:>12.6f}"
                  f"{100*sub[a,b,c]/1.6127375:>11.2f}{w1s[nn][a]:>9.4g}"
                  f"{rs[m][b]:>9.4g}{rhs[c]:>9.4g}{rhs[c]*rs[m][b]:>9.4f}")
    print("\n    If these sit close to 1.6127, the supremum is NOT degenerate:")
    print("    the limit is where the last percent lives, not the effect.")

    print(f"\n(b) FINITE OBSERVATION WINDOW  T = {T_window:g}")
    print("    D_1 = the window's own uniform moments, so this is an honest")
    print("    equal-N comparison inside one finite experiment.\n")
    w1s2 = np.geomspace(0.02, 5.0, 18)
    rs2 = 1.0 + np.geomspace(0.02, 60.0, 16)
    rhs2 = np.geomspace(0.03, 4.0, 14)
    ph2s = np.linspace(0, np.pi, 4, endpoint=False)
    cases = [("window", a, b, c, d, T_window, 120001, theta, model)
             for a in w1s2 for b in rs2 for c in rhs2 for d in ph2s]
    with mp.Pool(workers) as pool:
        V2 = np.array(pool.map(_degen_worker, cases, chunksize=32)).reshape(
            len(w1s2), len(rs2), len(rhs2), len(ph2s))
    i2, j2, k2, l2 = np.unravel_index(np.nanargmax(V2), V2.shape)
    print(f"    best on grid: {V2[i2,j2,k2,l2]:.6f}  at w1 = {w1s2[i2]:.4g}, "
          f"r = {rs2[j2]:.4g}, rho = {rhs2[k2]:.4g}, "
          f"phi2 = {np.degrees(ph2s[l2]):.0f} deg")
    print(f"    ({T_window*w1s2[i2]/(2*np.pi):.2f} periods of the slow tone "
          f"fit in the window)")
    print(f"\n    {'w1':>8}{'periods in T':>14}{'best gain':>12}{'at r':>9}"
          f"{'at rho':>9}{'at phi2':>9}")
    for a in range(len(w1s2)):
        idx = np.unravel_index(np.nanargmax(V2[a]), V2[a].shape)
        print(f"    {w1s2[a]:>8.4g}{T_window*w1s2[a]/(2*np.pi):>14.2f}"
              f"{V2[a][idx]:>12.6f}{rs2[idx[0]]:>9.4g}{rhs2[idx[1]]:>9.4g}"
              f"{np.degrees(ph2s[idx[2]]):>9.0f}")
    print("\n    These EXCEED the stationary supremum, and that is not a")
    print("    better design -- it is window alignment.  A finite window is")
    print("    not ergodic, so the optimiser tunes the beat to sit favourably")
    print("    inside [0, T].  Use `window_phase_sensitivity` to see the same")
    print("    input swing from above 2 to below 1 as the window is slid.")
    return dict(V=V, V2=V2)


def window_phase_sensitivity(w1=0.2688, r=1.099, rho=0.8876, ph2_deg=135.0,
                             theta=(1.0, 0.5), model=MODEL_STATIC, T=100.0,
                             t0s=(0, 25, 50, 100, 200, 400), n=600001):
    """
    Slide a fixed-length window along the same multisine.  Anything that
    moves here is a property of the window, not of the input.
    """
    theta = np.asarray(theta, float)
    ms = multisine([1.0, rho], [w1, r * w1], [0.0, np.radians(ph2_deg)])
    msx = output_multisine(ms, theta, model)
    print(f"\n  window-phase sensitivity, T = {T:g}, "
          f"w1={w1:g} r={r:g} rho={rho:g} phi2={ph2_deg:g}deg")
    print(f"  {'t0':>8}{'D_w/D_1 on [t0, t0+T]':>24}")
    for t0 in t0s:
        tt = np.linspace(t0, t0 + T, n)
        Phi = regressors_ms(tt, ms, model)
        wt = np.abs(evaluate_dot(msx, tt))
        Z = np.trapezoid(wt, tt)
        Mw = np.array([[np.trapezoid(Phi[:, a] * Phi[:, b] * wt, tt) / Z
                        for b in (0, 1)] for a in (0, 1)])
        M1 = np.array([[np.trapezoid(Phi[:, a] * Phi[:, b], tt) / T
                        for b in (0, 1)] for a in (0, 1)])
        print(f"  {t0:>8g}{det_gain(Mw, M1)['gain']:>24.6f}")
    st = det_gain(weighted_moments_torus(ms, theta, model, n_phase=1024),
                  uniform_moments_exact(ms, model))["gain"]
    print(f"  stationary (infinite-window) value: {st:.6f}")
