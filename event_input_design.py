"""
Input Design for Event-Based System Identification
Numerical experiments for Sections 3.1 - 3.3 of the assignment brief.

Structure
---------
  §3.1  classical input design: fixed sampling instants, design u
  §3.2  fixed u, swap periodic sampling for send-on-delta
  §3.3  make (A, omega) design variables under event-based sampling

Model
-----
  y(t_k) = phi^T(t_k) theta + e(t_k),      e ~ N(0, sigma^2) i.i.d.

  MODEL_STATIC   phi = [u, udot]                    (brief §3.1 verbatim)
  MODEL_DYNAMIC  phi = [G1(p)u, G2(p)u]             (brief §3.3, "stable
                 with G_j(s) = 1/(1+s T_j)           low-order system")

  Both are linear in theta and both have an analytic steady-state
  response to u(t) = A sin(omega t), so no ODE solver is needed.
  This matches the standard assumption that measurements are collected
  after the transient has decayed.

Event trigger (fixed, NOT a design variable -- see brief §2)
-----------------------------------------------------------
  tau_{i+1} = inf{ t > tau_i : |x(t) - x(tau_i)| >= Delta }
  where x(t) = phi^T(t) theta is the noise-free output.
"""

import numpy as np

# --------------------------------------------------------------------------
# Regressors
# --------------------------------------------------------------------------

MODEL_STATIC = "static"      # G(p) = theta1 + theta2 * p
MODEL_DYNAMIC = "dynamic"    # two first-order lags

TAU1, TAU2 = 0.5, 2.0        # time constants for MODEL_DYNAMIC


def regressors(t, A, omega, model=MODEL_STATIC):
    """phi(t) evaluated analytically at steady state. Returns (len(t), 2)."""
    if model == MODEL_STATIC:
        p1 = A * np.sin(omega * t)
        p2 = A * omega * np.cos(omega * t)
    elif model == MODEL_DYNAMIC:
        # G_j(j omega) = 1 / (1 + j omega T_j)
        cols = []
        for T in (TAU1, TAU2):
            G = 1.0 / (1.0 + 1j * omega * T)
            cols.append(A * np.abs(G) * np.sin(omega * t + np.angle(G)))
        p1, p2 = cols
    else:
        raise ValueError(f"unknown model: {model}")
    return np.column_stack([p1, p2])


def output(t, A, omega, theta, model=MODEL_STATIC):
    """Noise-free output x(t) = phi^T(t) theta."""
    return regressors(t, A, omega, model) @ theta


def output_envelope(A, omega, theta, model=MODEL_STATIC):
    """Amplitude C and phase psi of x(t) = C sin(omega t + psi)."""
    if model == MODEL_STATIC:
        z = A * (theta[0] + 1j * omega * theta[1])
    else:
        z = A * sum(th / (1.0 + 1j * omega * T)
                    for th, T in zip(theta, (TAU1, TAU2)))
    return np.abs(z), np.angle(z)


# --------------------------------------------------------------------------
# Sampling schemes
# --------------------------------------------------------------------------

def periodic_instants(T_total, N):
    """N uniformly spaced instants on [0, T_total)."""
    return np.arange(N) * (T_total / N)


def send_on_delta_instants(T_total, Delta, A, omega, theta,
                           model=MODEL_STATIC, oversample=4000):
    """
    Exact send-on-delta triggering.

    Walks a fine grid, and whenever |x(t) - x(tau_last)| first reaches
    Delta, linearly interpolates the crossing time.  The interpolation
    is what keeps this from being 'periodic sampling with extra steps' --
    event times land off the grid.
    """
    n_per_period = oversample
    n_grid = max(int(n_per_period * T_total * omega / (2 * np.pi)), 10_000)
    tg = np.linspace(0.0, T_total, n_grid)
    xg = output(tg, A, omega, theta, model)

    taus = [0.0]
    x_ref = xg[0]
    for k in range(1, n_grid):
        if abs(xg[k] - x_ref) >= Delta:
            # linear interpolation between grid points k-1 and k
            x_prev, x_cur = xg[k - 1], xg[k]
            target = x_ref + np.sign(x_cur - x_ref) * Delta
            denom = x_cur - x_prev
            frac = 0.0 if denom == 0 else (target - x_prev) / denom
            frac = min(max(frac, 0.0), 1.0)
            tau = tg[k - 1] + frac * (tg[k] - tg[k - 1])
            taus.append(tau)
            x_ref = target
    return np.asarray(taus[1:])          # drop the seeding instant


# --------------------------------------------------------------------------
# Information matrix and criteria
# --------------------------------------------------------------------------

def fisher(t_samples, A, omega, sigma, model=MODEL_STATIC):
    """I = (1/sigma^2) sum_k phi(t_k) phi^T(t_k)."""
    Phi = regressors(t_samples, A, omega, model)
    return (Phi.T @ Phi) / sigma ** 2


def criteria(I):
    """D (max det), A (min trace of inverse), E (max lambda_min)."""
    eig = np.linalg.eigvalsh(I)
    if eig.min() <= 1e-14 * max(eig.max(), 1e-30):
        return dict(det=np.linalg.det(I), trace_inv=np.inf, lam_min=eig.min())
    return dict(det=np.linalg.det(I),
                trace_inv=float(np.sum(1.0 / eig)),
                lam_min=float(eig.min()))


# --------------------------------------------------------------------------
# Monte Carlo check: does Cov(theta_hat) really match inv(I)?
# --------------------------------------------------------------------------

def monte_carlo_ls(t_samples, A, omega, theta_true, sigma,
                   model=MODEL_STATIC, n_runs=4000, seed=0):
    rng = np.random.default_rng(seed)
    Phi = regressors(t_samples, A, omega, model)
    x = Phi @ theta_true
    P = np.linalg.pinv(Phi.T @ Phi) @ Phi.T
    est = np.empty((n_runs, 2))
    for r in range(n_runs):
        est[r] = P @ (x + rng.normal(0.0, sigma, size=x.shape))
    return est.mean(axis=0), np.cov(est.T)


# --------------------------------------------------------------------------
# Experiment 1  --  §3.2, equal N, periodic vs event-based
# --------------------------------------------------------------------------

def experiment_equal_N(A=1.0, omega=1.0, theta=(1.0, 0.0), sigma=1.0,
                       n_periods=40, Delta=0.02, model=MODEL_STATIC):
    """
    Same continuous-time trajectory, two sampling schemes, SAME number of
    samples -- so the comparison isolates *where* you sample, not how often.
    """
    theta = np.asarray(theta, float)
    T_total = n_periods * 2 * np.pi / omega

    tau = send_on_delta_instants(T_total, Delta, A, omega, theta, model)
    N = len(tau)
    tk = periodic_instants(T_total, N)

    I_per = fisher(tk, A, omega, sigma, model)
    I_ev = fisher(tau, A, omega, sigma, model)
    return dict(N=N, tau=tau, tk=tk, I_per=I_per, I_ev=I_ev,
                c_per=criteria(I_per), c_ev=criteria(I_ev),
                T_total=T_total)


# --------------------------------------------------------------------------
# Experiment 2  --  is the 8/9 constant real?  sweep Delta
# --------------------------------------------------------------------------

def experiment_delta_sweep(A=1.0, omega=1.0, theta=(1.0, 0.0), sigma=1.0,
                           n_periods=40, model=MODEL_STATIC,
                           ratios=(0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005)):
    """
    The analytic 8/9 rests on replacing the event sum by an integral against
    the density dN/ds ∝ |cos s|.  That density ignores rounding at the
    turning points, so it should degrade as Delta grows toward C.
    """
    theta = np.asarray(theta, float)
    C, _ = output_envelope(A, omega, theta, model)
    rows = []
    for r in ratios:
        res = experiment_equal_N(A, omega, theta, sigma, n_periods,
                                 Delta=r * C, model=model)
        rows.append(dict(Delta_over_C=r,
                         N_per_period=res["N"] / n_periods,
                         ratio=res["c_ev"]["det"] / res["c_per"]["det"]))
    return rows


# --------------------------------------------------------------------------
# Experiment 3  --  §3.3, sweep the design variables (A, omega)
# --------------------------------------------------------------------------

def experiment_grid(theta=(1.0, 0.5), sigma=1.0, Delta=0.05,
                    A_vals=None, omega_vals=None, T_total=200.0,
                    model=MODEL_STATIC):
    """
    For each (A, omega): event count, information, and per-event information.
    T_total is held fixed -- so N is an *outcome*, not a setting.
    """
    theta = np.asarray(theta, float)
    if A_vals is None:
        A_vals = np.linspace(0.5, 3.0, 12)
    if omega_vals is None:
        omega_vals = np.linspace(0.4, 4.0, 12)

    shape = (len(A_vals), len(omega_vals))
    N = np.zeros(shape)
    detI = np.zeros(shape)
    lam = np.zeros(shape)
    ratio_diag = np.zeros(shape)      # I_22 / I_11 : where information goes

    for i, A in enumerate(A_vals):
        for j, w in enumerate(omega_vals):
            tau = send_on_delta_instants(T_total, Delta, A, w, theta, model)
            N[i, j] = len(tau)
            if len(tau) < 3:
                detI[i, j] = lam[i, j] = ratio_diag[i, j] = np.nan
                continue
            I = fisher(tau, A, w, sigma, model)
            c = criteria(I)
            detI[i, j] = c["det"]
            lam[i, j] = c["lam_min"]
            ratio_diag[i, j] = I[1, 1] / I[0, 0]
    return dict(A_vals=A_vals, omega_vals=omega_vals, N=N, detI=detI,
                lam_min=lam, ratio_diag=ratio_diag)


# --------------------------------------------------------------------------
# Experiment 4  --  §3.3 closing question
# --------------------------------------------------------------------------

def experiment_equivalent_inputs(theta=(1.0, 0.5), sigma=1.0, Delta=0.05,
                                 T_total=200.0, model=MODEL_STATIC,
                                 A_ref=1.0, omega_ref=1.0, omega_alt=2.0):
    """
    Two inputs that are INDISTINGUISHABLE under periodic sampling, compared
    under event-based sampling.

    For the static model det I_per / N^2 ∝ A^4 omega^2, so choosing
        A_alt = A_ref * sqrt(omega_ref / omega_alt)
    equalises them exactly.  Anything that differs afterwards is due to the
    event sequence alone -- the brief's claim that an excitation can be
    valuable "because it generates a more informative sequence of
    measurement events".
    """
    theta = np.asarray(theta, float)
    A_alt = A_ref * np.sqrt(omega_ref / omega_alt)

    out = []
    for A, w, tag in ((A_ref, omega_ref, "ref"), (A_alt, omega_alt, "alt")):
        tau = send_on_delta_instants(T_total, Delta, A, w, theta, model)
        N = len(tau)
        tk = periodic_instants(T_total, N)
        I_ev = fisher(tau, A, w, sigma, model)
        I_per = fisher(tk, A, w, sigma, model)
        C, psi = output_envelope(A, w, theta, model)
        out.append(dict(tag=tag, A=A, omega=w, N=N, psi_deg=np.degrees(psi),
                        det_per_n=criteria(I_per)["det"] / N ** 2,
                        det_ev_n=criteria(I_ev)["det"] / N ** 2,
                        lam_per_n=criteria(I_per)["lam_min"] / N,
                        lam_ev_n=criteria(I_ev)["lam_min"] / N,
                        split_per=I_per[1, 1] / I_per[0, 0],
                        split_ev=I_ev[1, 1] / I_ev[0, 0]))
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _fmt(x, w=12, p=4):
    return f"{x:>{w}.{p}g}"


def main():
    np.set_printoptions(precision=4, suppress=True)
    theta = np.array([1.0, 0.5])
    sigma = 1.0

    print("=" * 74)
    print("§3.2  SAME TRAJECTORY, SAME N -- periodic vs send-on-delta")
    print("=" * 74)
    res = experiment_equal_N(A=1.0, omega=1.0, theta=theta, sigma=sigma,
                             n_periods=40, Delta=0.02)
    C, psi = output_envelope(1.0, 1.0, theta)
    print(f"output envelope C = {C:.4f},  phase psi = {np.degrees(psi):.2f} deg")
    print(f"events N = {res['N']}   ({res['N']/40:.1f} per period; "
          f"predicted 4C/Delta = {4*C/0.02:.1f} per period)")
    print("\nI_periodic =\n", res["I_per"])
    print("I_event    =\n", res["I_ev"])
    print()
    print(f"{'criterion':<14}{'periodic':>12}{'event':>12}{'ev/per':>12}")
    for key, better in (("det", "max"), ("lam_min", "max"), ("trace_inv", "min")):
        p, e = res["c_per"][key], res["c_ev"][key]
        print(f"{key + ' (' + better + ')':<14}{_fmt(p)}{_fmt(e)}{_fmt(e/p)}")
    print("\nanalytic prediction for det ratio: 8/9 =", 8 / 9)

    print("\n" + "=" * 74)
    print("MONTE CARLO -- does Cov(theta_hat) match inv(I)?")
    print("=" * 74)
    for tag, ts in (("periodic", res["tk"]), ("event", res["tau"])):
        mean, cov = monte_carlo_ls(ts, 1.0, 1.0, theta, sigma, n_runs=4000)
        Iinv = np.linalg.inv(fisher(ts, 1.0, 1.0, sigma))
        print(f"\n{tag}:  mean(theta_hat) = {mean}   (true {theta})")
        print("  empirical cov =\n", cov)
        print("  inv(I)        =\n", Iinv)
        rel = np.linalg.norm(cov - Iinv) / np.linalg.norm(Iinv)
        print(f"  relative discrepancy (Frobenius) = {rel:.4f}")

    print("\n" + "=" * 74)
    print("DELTA SWEEP -- when does the |cos s| density approximation break?")
    print("=" * 74)
    print(f"{'Delta/C':>10}{'events/period':>16}{'det ratio':>14}"
          f"{'vs 8/9':>12}")
    for row in experiment_delta_sweep(A=1.0, omega=1.0, theta=theta,
                                      sigma=sigma, n_periods=40):
        print(f"{row['Delta_over_C']:>10.4g}{row['N_per_period']:>16.1f}"
              f"{row['ratio']:>14.4f}{row['ratio'] - 8/9:>12.4f}")

    print("\n" + "=" * 74)
    print("§3.3  DESIGN GRID over (A, omega),  T fixed, Delta fixed")
    print("=" * 74)
    g = experiment_grid(theta=theta, sigma=sigma, Delta=0.05,
                        A_vals=np.array([0.5, 1.0, 2.0, 4.0]),
                        omega_vals=np.array([0.5, 1.0, 2.0, 4.0]),
                        T_total=200.0)
    for name, M in (("event count N", g["N"]),
                    ("det I", g["detI"]),
                    ("det I / N^2  (per-event information)",
                     g["detI"] / g["N"] ** 2),
                    ("I_22 / I_11  (where information goes)",
                     g["ratio_diag"])):
        print(f"\n{name}")
        print("        " + "".join(f"{'w=' + str(w):>12}"
                                   for w in g['omega_vals']))
        for i, A in enumerate(g["A_vals"]):
            print(f"  A={A:<5}" + "".join(_fmt(v) for v in M[i]))

    print("\n" + "=" * 74)
    print("§3.3  TWO INPUTS EQUIVALENT UNDER PERIODIC SAMPLING")
    print("=" * 74)
    for mdl in (MODEL_STATIC, MODEL_DYNAMIC):
        rows = experiment_equivalent_inputs(theta=theta, sigma=sigma,
                                            Delta=0.05, T_total=200.0,
                                            model=mdl)
        print(f"\nmodel = {mdl}")
        print(f"{'tag':<6}{'A':>8}{'omega':>7}{'psi[deg]':>10}{'N':>8}"
              f"{'detI/N^2 per':>14}{'detI/N^2 ev':>13}"
              f"{'lam/N per':>11}{'lam/N ev':>10}"
              f"{'I22/I11 per':>13}{'I22/I11 ev':>12}")
        for r in rows:
            print(f"{r['tag']:<6}{r['A']:>8.4f}{r['omega']:>7.3g}"
                  f"{r['psi_deg']:>10.2f}{r['N']:>8d}"
                  f"{r['det_per_n']:>14.5g}{r['det_ev_n']:>13.5g}"
                  f"{r['lam_per_n']:>11.4g}{r['lam_ev_n']:>10.4g}"
                  f"{r['split_per']:>13.4g}{r['split_ev']:>12.4g}")


def make_plots(theta=(1.0, 0.5), sigma=1.0, outdir="."):
    """Four figures covering §3.2 and §3.3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta = np.asarray(theta, float)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) where the events land on the trajectory
    A, w, Delta = 1.0, 1.0, 0.15
    T2 = 2 * 2 * np.pi / w
    tg = np.linspace(0, T2, 3000)
    xg = output(tg, A, w, theta)
    tau = send_on_delta_instants(T2, Delta, A, w, theta)
    ax[0, 0].plot(tg, xg, lw=1.2, label="x(t)")
    ax[0, 0].plot(tau, output(tau, A, w, theta), "o", ms=5,
                  label=f"events (N={len(tau)})")
    C, _ = output_envelope(A, w, theta)
    for lev in np.arange(-C, C + Delta, Delta):
        ax[0, 0].axhline(lev, color="0.85", lw=0.5, zorder=0)
    ax[0, 0].set_title("§3.2  send-on-delta: uniform in OUTPUT, not time")
    ax[0, 0].set_xlabel("t"); ax[0, 0].set_ylabel("x(t)")
    ax[0, 0].legend(fontsize=8)

    # (b) inter-event time distribution
    A, w, Delta = 1.0, 1.0, 0.02
    T40 = 40 * 2 * np.pi / w
    tau = send_on_delta_instants(T40, Delta, A, w, theta)
    gaps = np.diff(tau)
    bins = np.logspace(np.log10(gaps.min()), np.log10(gaps.max()), 50)
    ax[0, 1].hist(gaps, bins=bins, color="C1", alpha=0.85)
    ax[0, 1].set_xscale("log")
    ax[0, 1].axvline(np.mean(gaps), color="k", ls="--", lw=1.2,
                     label=f"mean = {np.mean(gaps):.4f}\n(= periodic Ts)")
    ax[0, 1].set_title("§3.2  inter-event times: long tail from the peaks")
    ax[0, 1].set_xlabel("tau_{i+1} - tau_i"); ax[0, 1].set_ylabel("count")
    ax[0, 1].legend(fontsize=8)

    # (c) Delta sweep -> 8/9
    rows = experiment_delta_sweep(A=1.0, omega=1.0, theta=theta,
                                  sigma=sigma, n_periods=40)
    x = [r["N_per_period"] for r in rows]
    y = [r["ratio"] for r in rows]
    ax[1, 0].semilogx(x, y, "o-", label="numerical")
    ax[1, 0].axhline(8 / 9, color="r", ls="--", label="analytic 8/9")
    ax[1, 0].set_title("det I_event / det I_periodic  at equal N")
    ax[1, 0].set_xlabel("events per period  (= 4C/Delta)")
    ax[1, 0].set_ylabel("ratio"); ax[1, 0].legend(fontsize=8)

    # (d) where information goes, vs frequency
    ws = np.linspace(0.3, 5.0, 24)
    split_ev, split_per = [], []
    for w in ws:
        Tw = 40 * 2 * np.pi / w
        Cw, _ = output_envelope(1.0, w, theta)
        tau = send_on_delta_instants(Tw, 0.02 * Cw, 1.0, w, theta)
        tk = periodic_instants(Tw, len(tau))
        Ie = fisher(tau, 1.0, w, sigma)
        Ip = fisher(tk, 1.0, w, sigma)
        split_ev.append(Ie[1, 1] / Ie[0, 0])
        split_per.append(Ip[1, 1] / Ip[0, 0])
    ax[1, 1].loglog(ws, split_per, "o-", label="periodic  (= omega^2)")
    ax[1, 1].loglog(ws, split_ev, "s-", label="event-based")
    ax[1, 1].set_title("§3.3  I22/I11: how information splits between params")
    ax[1, 1].set_xlabel("omega"); ax[1, 1].set_ylabel("I22 / I11")
    ax[1, 1].legend(fontsize=8)

    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout()
    path = f"{outdir}/event_input_design_figures.png"
    fig.savefig(path, dpi=130)
    print(f"\nfigures written to {path}")


if __name__ == "__main__":
    main()
    make_plots()
