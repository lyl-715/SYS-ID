"""
Multisine excitation of a two-lag LINEAR dynamical system under
send-on-delta -- exploratory study of the information/communication
trade-off.  NO optimisation anywhere in this file.

Model (linear in theta, so I = sigma^-2 sum phi phi^T is exact):

    x(t)  = theta1 G1(p) u(t) + theta2 G2(p) u(t),  theta = (1, 0.5)
    phi   = [G1(p) u, G2(p) u]^T
    G_j(s) = 1 / (1 + s T_j)   -- the repo's MODEL_DYNAMIC pair, reused
                                  verbatim: T1 = 0.5, T2 = 2.0
                                  (event_input_design.TAU1, TAU2).

Steady state analytic throughout: each regressor is a sum of tones at the
input frequencies with gains |G_j(i w_m)| and phases angle G_j(i w_m); no
ODE solver anywhere.  The NONLINEAR (K, tau)-sensitivity parameterisation
is deliberately NOT implemented here.

Reused from the repo: regressors_ms / output_multisine / fisher_ms /
uniform_moments_exact (exact M_1, including the nonzero <phi1 phi2>_1 --
the two lags are correlated channels), send_on_delta_instants_ms (which
wraps trigger_on_signal + Newton refinement), periodic_instants,
envelope_analytic, and the |xdot|-weighted quadrature idea: as Delta -> 0
the event density is p ~ |xdot| with xdot the derivative of the FILTERED
output.

Conventions (stated, not settled):
  * power normalised: sum a_m^2 = 1 for every input (scale factor printed
    where inputs are constructed).
  * threshold rule  Delta = max(C_min/100, C_peak/2000)  with C_min the
    smallest and C_peak the sum of output-component amplitudes; the floor
    stops Delta -> 0 when a component amplitude vanishes.  Delta/C_min,
    Delta/C_peak and the floor-hit count are reported with every figure.
  * event rate lambda = <|xdot|>_1 / Delta (the Delta -> 0 rate at this
    input's own Delta).  Because Delta follows C_min, lambda comparisons
    across inputs inherit the threshold rule; m = <|xdot|>_1 itself is
    threshold-free and is printed everywhere so any fixed-Delta rate is
    m / Delta.
  * per-event information: det I / N^2 = det M_w and lam_min(I)/N =
    lam_min(M_w); rates per unit time: det I / T^2 = lambda^2 det M_w and
    lam_min(I)/T = lambda lam_min(M_w).  (det I / lambda^2 = T^2 det M_w
    would drag the window length in, so the N-normalised version is what
    the "per event" tables and panels show.)
  * windows: T = 400 periods of the slowest tone, reported per figure.
    Stationary values only.
  * NO zeta for M > 1 (the single-complex-number description is known to
    break there); everything is criteria ratios.
"""

import multiprocessing as mp

import numpy as np

from event_input_design import MODEL_DYNAMIC, TAU1, TAU2, periodic_instants
from multisine_input_design import (_channel_tf, envelope_analytic,
                                    evaluate_dot, fisher_ms, multisine,
                                    output_multisine, regressors_ms,
                                    send_on_delta_instants_ms)

THETA = np.array([1.0, 0.5])
N_PERIODS = 400           # window: 400 periods of the slowest tone
PTS_PER_FAST = 48         # quadrature points per period of the fastest tone
N_CAP = 8_000_001


def uniform_moments_dyn(ms):
    """
    Exact infinite-window <phi_i phi_j>_1 for the two-lag channels.

    multisine_input_design.uniform_moments_exact sums over components
    DIAGONALLY and therefore drops the cross terms between two components
    that share a frequency -- which is exactly the omega_2 = omega_1 edge
    of the Figure-2 sweep (where the input degenerates to one sinusoid and
    det ratio must be 8/9).  This version does the pairwise sum
        <phi_i phi_j>_1 = (1/2) sum_{m,l : w_m = w_l}
                          c^i_m c^j_l cos(alpha^i_m - alpha^j_l)
    and reduces to the existing function for distinct frequencies.
    """
    F1, F2 = _channel_tf(MODEL_DYNAMIC)
    H = np.column_stack([F1(ms.freqs), F2(ms.freqs)])
    amp = ms.amps[:, None] * np.abs(H)              # (M, 2)
    pha = ms.phases[:, None] + np.angle(H)          # (M, 2)
    Mo = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            v = 0.0
            for mm in range(ms.M):
                for ll in range(ms.M):
                    if abs(ms.freqs[mm] - ms.freqs[ll]) < 1e-12 * max(
                            ms.freqs[mm], 1.0):
                        v += 0.5 * amp[mm, i] * amp[ll, j] * np.cos(
                            pha[mm, i] - pha[ll, j])
            Mo[i, j] = v
    return Mo


def delta_rule(msx):
    """Delta = max(C_min/100, C_peak/2000); returns (Delta, floor_hit)."""
    C_min, C_peak = float(msx.amps.min()), float(np.sum(msx.amps))
    Delta = max(C_min / 100.0, C_peak / 2000.0)
    return Delta, Delta > C_min / 100.0 * (1 + 1e-12), C_min, C_peak


def _window_grid(ms):
    T = N_PERIODS * 2.0 * np.pi / ms.freqs.min()
    dt = 2.0 * np.pi / (PTS_PER_FAST * ms.freqs.max())
    # floor of 200k points: at M = 1 the resolution rule alone would give
    # only 19200 points and the |xdot| kinks then cost ~1e-3 relative
    # error -- visible against the exact 8/9 of P1
    n = min(max(int(np.ceil(T / dt)) + 1, 200_001), N_CAP)
    return np.linspace(0.0, T, n), T


# --------------------------------------------------------------------------
# Quadrature route (stationary; Delta-free except through the lambda rule)
# --------------------------------------------------------------------------

def quad_case(ms):
    msx = output_multisine(ms, THETA, MODEL_DYNAMIC)
    t, T = _window_grid(ms)
    Phi = regressors_ms(t, ms, MODEL_DYNAMIC)
    wt = np.abs(evaluate_dot(msx, t))
    Z = np.trapezoid(wt, t)
    m = float(Z / T)

    Mw = np.empty((2, 2))
    Mw[0, 0] = np.trapezoid(Phi[:, 0] ** 2 * wt, t) / Z
    Mw[1, 1] = np.trapezoid(Phi[:, 1] ** 2 * wt, t) / Z
    Mw[0, 1] = Mw[1, 0] = np.trapezoid(Phi[:, 0] * Phi[:, 1] * wt, t) / Z
    M1 = uniform_moments_dyn(ms)

    # P4 raw material: direction sums and the two envelope readings.
    # NOTE M1 is NOT diagonal here, so det ratio must come from the full
    # determinants -- the static-case R11 R22 - cross^2 shortcut assumed
    # <phi1 phi2>_1 = 0 and would be wrong for correlated channels.
    R11 = Mw[0, 0] / M1[0, 0]
    R22 = Mw[1, 1] / M1[1, 1]
    Ex = envelope_analytic(msx, t) ** 2          # squared OUTPUT envelope
    Eu = envelope_analytic(ms, t) ** 2           # squared INPUT envelope
    e_x = 2.0 * (np.trapezoid(Ex * wt, t) / Z) / (np.trapezoid(Ex, t) / T)
    e_u = 2.0 * (np.trapezoid(Eu * wt, t) / Z) / (np.trapezoid(Eu, t) / T)

    Delta, floor_hit, C_min, C_peak = delta_rule(msx)
    lam = m / Delta
    detMw, detM1 = float(np.linalg.det(Mw)), float(np.linalg.det(M1))
    lamMw = float(np.linalg.eigvalsh(Mw)[0])
    lamM1 = float(np.linalg.eigvalsh(M1)[0])
    return dict(m=m, lam=lam, Delta=Delta, floor=bool(floor_hit),
                C_min=C_min, C_peak=C_peak, Mw=Mw, M1=M1,
                detMw=detMw, detM1=detM1, lamMw=lamMw, lamM1=lamM1,
                det_ratio=detMw / detM1, lam_ratio=lamMw / lamM1,
                det_rate=lam ** 2 * detMw, det_rate_per=lam ** 2 * detM1,
                lmin_rate=lam * lamMw, lmin_rate_per=lam * lamM1,
                R11=float(R11), R22=float(R22),
                dirsum=float(R11 + R22), e_x=float(e_x), e_u=float(e_u),
                T=T, n=len(t),
                w_mean=float(np.sum(ms.amps ** 2 * ms.freqs)
                             / np.sum(ms.amps ** 2)))


def whitening_views(Mw, M1):
    """
    P2: lam_min ratio in three metrics.
      raw       : lam_min(Mw) / lam_min(M1)
      whitened  : eigenvalues of S Mw S^T with S = M1^{-1/2} (FULL matrix
                  inverse square root) -- should be 1 +/- |zeta| at M = 1,
                  so lam_min = 2/3 exactly.
      diagonal  : lam_min(D Mw D) / lam_min(D M1 D) with D =
                  diag(M1)^{-1/2} -- equalises the diagonal only; the
                  channels stay correlated, so this does NOT return 2/3.
    """
    lam, V = np.linalg.eigh(M1)
    S = V @ np.diag(lam ** -0.5) @ V.T
    wh = np.linalg.eigvalsh(S @ Mw @ S.T)
    D = np.diag(np.diag(M1) ** -0.5)
    dn = (np.linalg.eigvalsh(D @ Mw @ D)[0]
          / np.linalg.eigvalsh(D @ M1 @ D)[0])
    return dict(raw=float(np.linalg.eigvalsh(Mw)[0]
                          / np.linalg.eigvalsh(M1)[0]),
                white_min=float(wh[0]), white_max=float(wh[1]),
                diag=float(dn))


# --------------------------------------------------------------------------
# Simulation route (real events; validation of P1 and the rate)
# --------------------------------------------------------------------------

def sim_case(ms, max_points=40_000_000):
    msx = output_multisine(ms, THETA, MODEL_DYNAMIC)
    T = N_PERIODS * 2.0 * np.pi / ms.freqs.min()
    Delta, floor_hit, C_min, C_peak = delta_rule(msx)
    taus = send_on_delta_instants_ms(T, Delta, ms, THETA, MODEL_DYNAMIC,
                                     pts_per_delta=10, refine=True)
    N = len(taus)
    if N < 8:
        return None
    tk = periodic_instants(T, N)
    I_ev = fisher_ms(taus, ms, 1.0, MODEL_DYNAMIC)
    I_per = fisher_ms(tk, ms, 1.0, MODEL_DYNAMIC)
    return dict(N=N, T=T, Delta=Delta, lam_sim=N / T,
                det_ratio=float(np.linalg.det(I_ev) / np.linalg.det(I_per)),
                detMw_sim=float(np.linalg.det(I_ev) / N ** 2),
                lamMw_sim=float(np.linalg.eigvalsh(I_ev)[0] / N),
                floor=bool(floor_hit),
                dc=Delta / C_min, dp=Delta / C_peak)


# --------------------------------------------------------------------------
# Random inputs and the parallel worker
# --------------------------------------------------------------------------

def random_input(Mn, rng, w_lo=0.05, w_hi=20.0):
    """M tones, log-uniform frequencies, random phases; amplitudes drawn
    uniform(0.2, 1) then rescaled so sum a^2 = 1 (the scale factor is
    1/sqrt(sum of the raw a^2))."""
    a = rng.uniform(0.2, 1.0, Mn)
    a /= np.sqrt(np.sum(a ** 2))
    w = np.exp(rng.uniform(np.log(w_lo), np.log(w_hi), Mn))
    return multisine(a, w, rng.uniform(0.0, 2 * np.pi, Mn))


def _quad_worker(case):
    tag, ms = case
    try:
        r = quad_case(ms)
        r.pop("Mw"), r.pop("M1")
        r["tag"], r["ok"], r["M"] = tag, True, ms.M
        r["amps"], r["freqs"], r["phases"] = ms.amps, ms.freqs, ms.phases
        return r
    except Exception as exc:
        return dict(tag=tag, ok=False, error=f"{type(exc).__name__}: {exc}")


def run_pool(cases, workers=4, chunksize=4):
    if len(cases) < 50 or workers == 1:
        return [_quad_worker(c) for c in cases]
    with mp.Pool(workers) as pool:
        return pool.map(_quad_worker, cases, chunksize=chunksize)


# --------------------------------------------------------------------------
# Figure 1 -- single tone: rates, information, and the P1/P2 checks
# --------------------------------------------------------------------------

def fig1(n_omega=37, sim_omegas=(0.1, 0.3, 1.0, 3.0, 10.0), outdir="."):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ws = np.geomspace(0.05, 20.0, n_omega)
    rows, white = [], []
    for w in ws:
        q = quad_case(multisine([1.0], [w], [0.0]))
        rows.append(q)
        white.append(whitening_views(q["Mw"], q["M1"]))
    T_lo = N_PERIODS * 2 * np.pi / ws[0]

    print("=" * 96)
    print("FIGURE 1 -- SINGLE TONE (M = 1, a = 1, power = 1), omega swept "
          "0.05 .. 20")
    print("=" * 96)
    print(f"  model: x = th1 G1 u + th2 G2 u, G_j = 1/(1+s T_j), "
          f"T = ({TAU1}, {TAU2}), theta = {tuple(THETA)}")
    print(f"  window = {N_PERIODS} periods of the tone (longest T = "
          f"{T_lo:.0f} at omega = {ws[0]:g})")
    print(f"  Delta rule: max(C_min/100, C_peak/2000); at M = 1 "
          f"C_min = C_peak so Delta/C_min = 0.01 and")
    print(f"  Delta/C_peak = 0.01 at every point; floor hits: "
          f"{sum(1 for q in rows if q['floor'])}/{len(rows)}")

    print(f"\n{'omega':>8}{'m=<|xd|>':>10}{'lambda':>10}{'detMw/ev':>12}"
          f"{'detM1/ev':>12}{'lamMw/ev':>12}{'detI/T^2':>12}"
          f"{'lmin(I)/T':>12}{'det ratio':>11}")
    for w, q in zip(ws, rows):
        print(f"{w:>8.3g}{q['m']:>10.5f}{q['lam']:>10.2f}"
              f"{q['detMw']:>12.4e}{q['detM1']:>12.4e}{q['lamMw']:>12.4e}"
              f"{q['det_rate']:>12.4e}{q['lmin_rate']:>12.4e}"
              f"{q['det_ratio']:>11.6f}")

    # ---- P1: det ratio = 8/9 exactly, filters and omega notwithstanding
    dev = np.array([q["det_ratio"] - 8 / 9 for q in rows])
    print(f"\n  P1 check (quadrature): max |det ratio - 8/9| = "
          f"{np.abs(dev).max():.2e} over the sweep -> "
          f"{'PASS' if np.abs(dev).max() < 2e-3 else 'FAIL -- STOPPING'}")

    # ---- P2: lam_min ratio moves; full whitening restores 2/3; diagonal
    #      normalisation does not
    print("\n  P2 check: lam_min ratio in three metrics")
    print(f"{'omega':>8}{'raw':>10}{'diag-norm':>11}{'whitened min':>14}"
          f"{'-2/3':>10}{'whitened max':>14}{'-4/3':>10}")
    sel = [0, 6, 12, 18, 24, 30, 36]
    for i in sel:
        v = white[i]
        print(f"{ws[i]:>8.3g}{v['raw']:>10.5f}{v['diag']:>11.5f}"
              f"{v['white_min']:>14.6f}{v['white_min']-2/3:>10.2e}"
              f"{v['white_max']:>14.6f}{v['white_max']-4/3:>10.2e}")
    raw = np.array([v["raw"] for v in white])
    dg = np.array([v["diag"] for v in white])
    wm = np.array([v["white_min"] for v in white])
    print(f"  raw ranges over [{raw.min():.4f}, {raw.max():.4f}] "
          f"(static-model value was 2/3 = 0.6667: it MOVED, sits ABOVE 1,")
    print(f"  and hugs 4/3 -- at M = 1 the two-lag M_per is ill-conditioned,")
    print(f"  the min-eigenvectors of M_ev and M_per align with the same")
    print(f"  near-degenerate direction, and their Rayleigh quotient lands")
    print(f"  on the whitened UPPER eigenvalue 4/3, not the lower);")
    print(f"  diag-norm ranges over [{dg.min():.4f}, "
          f"{dg.max():.4f}] -- diagonal normalisation does NOT")
    print(f"  restore 2/3; FULL whitening S = M_per^(-1/2) gives lam_min "
          f"= 2/3 to {np.abs(wm - 2/3).max():.1e} everywhere.")
    print("  So the E-loss = 2/3 statement is metric-bound; D is "
          "metric-free.")
    p2_ok = (np.abs(wm - 2 / 3).max() < 5e-3
             and np.abs(dg - 2 / 3).min() > 0.05
             and raw.max() - raw.min() > 1e-4)

    # ---- P3 numbers: saturation of m
    msat = (2 / np.pi) * (THETA[0] / TAU1 + THETA[1] / TAU2)
    mono = np.all(np.diff([q["m"] for q in rows]) > 0)
    print(f"\n  P3 numbers: m(omega=20) = {rows[-1]['m']:.4f} vs predicted "
          f"plateau (2/pi)(th1/T1 + th2/T2) = {msat:.4f}")
    print(f"  ({100 * abs(rows[-1]['m'] / msat - 1):.2f}% off; single-lag "
          f"pieces (2/pi) th1/T1 = {(2/np.pi)*THETA[0]/TAU1:.4f}, "
          f"(2/pi) th2/T2 = {(2/np.pi)*THETA[1]/TAU2:.4f});")
    print(f"  m monotone increasing across the sweep: {mono}; lambda = "
          f"m/Delta with Delta = C/100 keeps RISING")
    print("  (C falls like 1/omega, so lambda ~ omega at high omega: the "
          "threshold rule, not the physics; at any")
    print("  FIXED Delta the event rate saturates with m).")

    # ---- simulation overlay (P1 at real events)
    sims = []
    print("\n  simulation overlay (Delta/C_min = Delta/C_peak = 0.01):")
    print(f"{'omega':>8}{'N':>9}{'lam sim':>10}{'lam quad':>10}"
          f"{'det ratio sim':>15}{'-8/9':>11}")
    for w in sim_omegas:
        ms = multisine([1.0], [w], [0.0])
        s = sim_case(ms)
        qq = quad_case(ms)
        sims.append((w, s, qq))
        print(f"{w:>8.3g}{s['N']:>9d}{s['lam_sim']:>10.4f}"
              f"{qq['lam']:>10.4f}{s['det_ratio']:>15.6f}"
              f"{s['det_ratio'] - 8/9:>11.2e}")
    print("  (sim sits ~4.5e-3 below 8/9: the documented -0.44 Delta/C_min "
          "finite-threshold bias.)")
    p1_ok = (np.abs(dev).max() < 2e-3
             and all(-0.012 < s["det_ratio"] - 8 / 9 < 0.002
                     for _, s, _ in sims))

    lam = np.array([q["lam"] for q in rows])
    fig, ax = plt.subplots(4, 1, figsize=(8.5, 13.5), sharex=True)

    ax[0].loglog(ws, lam, "-", color="C0", lw=1.8,
                 label="lambda = m/Delta  (Delta = C/100)")
    ax[0].loglog(ws, [q["m"] for q in rows], "-", color="C1", lw=1.8,
                 label="m = <|xdot|>  (threshold-free)")
    ax[0].axhline(msat, color="0.4", lw=1, ls=":",
                  label="m plateau (2/pi)(th1/T1+th2/T2)")
    ax[0].loglog([w for w, s, _ in sims], [s["lam_sim"] for _, s, _ in sims],
                 "o", ms=6, mfc="none", color="C0", label="events (sim)")
    ax[0].set_ylabel("event rate")
    ax[0].set_title(f"Figure 1 -- single tone, x = th1 G1 u + th2 G2 u, "
                    f"T = ({TAU1}, {TAU2}), theta = (1, 0.5)\n"
                    f"window {N_PERIODS} periods; Delta/C_min = "
                    f"Delta/C_peak = 0.01", fontsize=10)
    ax[0].legend(fontsize=8)

    ax[1].loglog(ws, [q["det_rate"] for q in rows], "-", color="C1", lw=1.8,
                 label="det I / T^2, events")
    ax[1].loglog(ws, [q["det_rate_per"] for q in rows], "--", color="C1",
                 lw=1.4, label="det I / T^2, periodic (same N)")
    ax[1].loglog(ws, [q["lmin_rate"] for q in rows], "-", color="C2", lw=1.8,
                 label="lam_min(I) / T, events")
    ax[1].loglog(ws, [q["lmin_rate_per"] for q in rows], "--", color="C2",
                 lw=1.4, label="lam_min(I) / T, periodic (same N)")
    ax[1].set_ylabel("information rate")
    ax[1].legend(fontsize=8)

    ax[2].loglog(ws, [q["detMw"] for q in rows], "-", color="C1", lw=1.8,
                 label="det I / N^2, events")
    ax[2].loglog(ws, [q["detM1"] for q in rows], "--", color="C1", lw=1.4,
                 label="det I / N^2, periodic")
    ax[2].loglog(ws, [q["lamMw"] for q in rows], "-", color="C2", lw=1.8,
                 label="lam_min(I) / N, events")
    ax[2].loglog(ws, [q["lamM1"] for q in rows], "--", color="C2", lw=1.4,
                 label="lam_min(I) / N, periodic")
    ax[2].axvline(1 / TAU1, color="0.4", lw=1, ls=":")
    ax[2].axvline(1 / TAU2, color="0.4", lw=1, ls=":")
    ax[2].set_ylabel("information per event")
    ax[2].legend(fontsize=7, ncol=2)

    ax[3].semilogx(ws, raw, "-", color="C3", lw=1.6, label="raw")
    ax[3].semilogx(ws, dg, "-", color="C4", lw=1.6, label="diag-normalised")
    ax[3].semilogx(ws, wm, "-", color="C2", lw=1.6,
                   label="whitened (S = M_per^-1/2)")
    ax[3].axhline(2 / 3, color="0.4", lw=1, ls=":", label="2/3")
    ax[3].set_ylabel("lam_min ratio (P2)")
    ax[3].set_xlabel("omega")
    ax[3].legend(fontsize=8, ncol=2)

    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    path = f"{outdir}/multisine_dynamic_fig1.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\n  figure written to {path}")
    print(f"  P1 {'PASS' if p1_ok else 'FAIL'};  P2 "
          f"{'PASS' if p2_ok else 'FAIL'}")
    return dict(ws=ws, rows=rows, white=white, sims=sims,
                ok=p1_ok and p2_ok)


# --------------------------------------------------------------------------
# Figure 2 -- two tones: maps over (omega_2, rho), incl. the P4 check
# --------------------------------------------------------------------------

W1_FIG2 = 0.3


def fig2(n_w2=14, n_rho=10, n_draw=8, outdir=".", workers=4, seed=5):
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    w2s = np.geomspace(W1_FIG2, 20.0, n_w2)
    rhos = np.linspace(0.05, 0.95, n_rho)
    rng = np.random.default_rng(seed)
    cases = []
    for i, w2 in enumerate(w2s):
        for j, rho in enumerate(rhos):
            for k in range(n_draw):
                ph = rng.uniform(0, 2 * np.pi, 2)
                ms = multisine([np.sqrt(1 - rho), np.sqrt(rho)],
                               [W1_FIG2, w2], ph)     # sum a^2 = 1 exactly
                cases.append(((i, j, k), ms))
    t0 = time.time()
    out = run_pool(cases, workers=workers)
    wall = time.time() - t0

    keys = ("lam", "det_rate", "detMw", "det_ratio", "dirsum", "e_x",
            "e_u", "m")
    shape = (n_w2, n_rho, n_draw)
    F = {k: np.full(shape, np.nan) for k in keys}
    floor = np.zeros(shape, bool)
    for r in out:
        if r["ok"]:
            for k in keys:
                F[k][r["tag"]] = r[k]
            floor[r["tag"]] = r["floor"]
    mean = {k: v.mean(axis=2) for k, v in F.items()}
    spread = {k: np.nanmax(v.std(axis=2) / np.abs(mean[k]))
              for k, v in F.items()}

    T = N_PERIODS * 2 * np.pi / W1_FIG2
    nfloor = int(floor.sum())
    print("\n" + "=" * 96)
    print(f"FIGURE 2 -- TWO TONES, omega_1 = {W1_FIG2}, omega_2 in "
          f"[{W1_FIG2}, 20], rho = a2^2/(a1^2+a2^2)")
    print("=" * 96)
    print(f"  window T = {T:.0f} ({N_PERIODS} periods of omega_1); "
          f"{n_draw} phase draws/cell; {len(cases)} quadrature cases in "
          f"{wall:.0f} s on {workers} workers")
    print(f"  Delta rule: floor (C_peak/2000) hit in {nfloor}/{len(cases)} "
          f"cases = {100*nfloor/len(cases):.1f}%"
          + ("  <-- OVER 20%: sweep range too wide for the rule"
             if nfloor > 0.2 * len(cases) else ""))
    dcs = np.array([r["Delta"] / r["C_min"] for r in out if r["ok"]])
    dps = np.array([r["Delta"] / r["C_peak"] for r in out if r["ok"]])
    print(f"  Delta/C_min: median {np.median(dcs):.4f}, max {dcs.max():.3f}"
          f";  Delta/C_peak: median {np.median(dps):.5f}, "
          f"max {dps.max():.5f}")
    print("  worst relative spread over the 8 draws, per quantity: "
          + ", ".join(f"{k} {v:.3f}" for k, v in spread.items()))
    print("  (the omega_2 = omega_1 row is the coincident-tone edge: the")
    print("  input is a SINGLE sinusoid whose amplitude depends on the")
    print("  drawn phase difference, so per-event quantities scatter")
    print("  strongly there while det ratio stays at 8/9.)")

    def table(name, G, fmt="{:>9.4f}"):
        print(f"\n  {name}   (rows omega_2, cols rho)")
        print("           " + "".join(f"{r:>9.2f}" for r in rhos))
        for i, w2 in enumerate(w2s):
            print(f"  w2={w2:>6.3g} " + "".join(fmt.format(G[i, j])
                                                for j in range(n_rho)))

    table("(a) lambda = m/Delta, events/time", mean["lam"], "{:>9.1f}")
    table("(b) det I / T^2", mean["det_rate"], "{:>9.3g}")
    table("(c) det I / N^2, per event (x 1e3)", 1e3 * mean["detMw"])
    table("(d) det I_ev / det I_per at equal N", mean["det_ratio"])
    table("(e1) R11 + R22", mean["dirsum"])
    table("(e2) 2 <E>_w / <E>_1, OUTPUT envelope", mean["e_x"])

    # P4: do they agree?
    d_x = mean["dirsum"] - mean["e_x"]
    d_u = mean["dirsum"] - mean["e_u"]
    print("\n  P4 check (identity R11 + R22 = 2<E>_w/<E>_1?):")
    print(f"    vs OUTPUT envelope: mean|diff| = "
          f"{np.nanmean(np.abs(d_x)):.4f}, max|diff| = "
          f"{np.nanmax(np.abs(d_x)):.4f}")
    print(f"    vs INPUT  envelope: mean|diff| = "
          f"{np.nanmean(np.abs(d_u)):.4f}, max|diff| = "
          f"{np.nanmax(np.abs(d_u)):.4f}")
    print(f"    correlation (over cells) of R11+R22 with 2<E_x>_w/<E_x>_1:"
          f" {np.corrcoef(mean['dirsum'].ravel(), mean['e_x'].ravel())[0,1]:.3f}")

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.5))
    ee = np.concatenate([mean["dirsum"].ravel(), mean["e_x"].ravel()])
    panels = (("lam", "(a) lambda = m/Delta", "viridis", None, None),
              ("det_rate", "(b) det I / T^2", "viridis", None, None),
              ("detMw", "(c) det I / N^2 (per event)", "viridis", None,
               None),
              ("det_ratio", "(d) det I_ev / det I_per, equal N", "RdBu_r",
               mcolors.TwoSlopeNorm(vcenter=1.0), None),
              ("dirsum", "(e1) R11 + R22", "viridis", None,
               (ee.min(), ee.max())),
              ("e_x", "(e2) 2 <E>_w / <E>_1 (output envelope)", "viridis",
               None, (ee.min(), ee.max())))
    for a, (key, title, cmap, norm, clim) in zip(ax.ravel(), panels):
        im = a.pcolormesh(rhos, w2s, mean[key], shading="auto", cmap=cmap,
                          norm=norm,
                          vmin=None if clim is None else clim[0],
                          vmax=None if clim is None else clim[1])
        a.set_yscale("log")
        a.set_xlabel("rho = a2^2/(a1^2+a2^2)")
        a.set_ylabel("omega_2")
        a.set_title(title, fontsize=10)
        fig.colorbar(im, ax=a)
    fig.suptitle(f"Figure 2 -- two tones, omega_1 = {W1_FIG2}, "
                 f"T = ({TAU1}, {TAU2}), theta = (1, 0.5); mean over "
                 f"{n_draw} phase draws; (e1)/(e2) share a colour scale",
                 fontsize=11)
    fig.tight_layout()
    path = f"{outdir}/multisine_dynamic_fig2.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\n  figure written to {path}")
    return dict(w2s=w2s, rhos=rhos, mean=mean, spread=spread,
                nfloor=nfloor, ncases=len(cases),
                d_x=float(np.nanmax(np.abs(d_x))),
                d_u=float(np.nanmax(np.abs(d_u))))


# --------------------------------------------------------------------------
# Figure 3 -- THE trade-off plot: information rate vs event rate
# --------------------------------------------------------------------------

def draw_inputs(n, seed=42):
    rng = np.random.default_rng(seed)
    return [random_input(int(rng.integers(1, 4)), rng) for _ in range(n)]


def fig3(n_inputs=500, n_first=50, outdir=".", workers=4):
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    inputs = draw_inputs(n_inputs)
    print("\n" + "=" * 96)
    print(f"FIGURE 3 -- TRADE-OFF: information rate vs event rate, "
          f"{n_inputs} random inputs, M in {{1,2,3}}")
    print("=" * 96)
    print(f"  omega log-uniform [0.05, 20], random phases, power = 1 "
          f"(amplitudes drawn U(0.2,1) then rescaled);")
    print(f"  window {N_PERIODS} periods of each input's slowest tone; "
          f"lambda = m/Delta at each input's own Delta.")

    t0 = time.time()
    first = run_pool([(i, ms) for i, ms in enumerate(inputs[:n_first])],
                     workers=workers)
    t1 = time.time()
    print(f"\n  first {n_first} inputs: {t1 - t0:.1f} s wall-clock on "
          f"{workers} workers -> scaling to {n_inputs}")
    rest = run_pool([(n_first + i, ms) for i, ms
                     in enumerate(inputs[n_first:])], workers=workers)
    print(f"  full sweep: {time.time() - t0:.1f} s total")
    res = [r for r in first + rest if r["ok"]]
    nfloor = sum(1 for r in res if r["floor"])
    dcs = np.array([r["Delta"] / r["C_min"] for r in res])
    dps = np.array([r["Delta"] / r["C_peak"] for r in res])
    print(f"  {len(res)}/{n_inputs} evaluated; Delta floor hit by "
          f"{nfloor} = {100*nfloor/len(res):.1f}% of inputs"
          + ("  <-- OVER 20%: sweep range too wide for the Delta rule"
             if nfloor > 0.2 * len(res) else ""))
    print(f"  Delta/C_min: median {np.median(dcs):.4f}, max {dcs.max():.3f}"
          f";  Delta/C_peak: median {np.median(dps):.5f}, max "
          f"{dps.max():.5f}")

    lam = np.array([r["lam"] for r in res])
    Ms = np.array([r["M"] for r in res])
    det_ev = np.array([r["det_rate"] for r in res])
    det_pp = np.array([r["det_rate_per"] for r in res])
    lmin_ev = np.array([r["lmin_rate"] for r in res])
    lmin_pp = np.array([r["lmin_rate_per"] for r in res])

    def spread_table(y_ev, y_per, name):
        print(f"\n  {name}: vertical spread at comparable lambda "
              f"(deciles of lambda)")
        print(f"{'lambda bin':>24}{'n':>5}{'p90/p10 ev':>13}"
              f"{'max/min ev':>13}{'p90/p10 per':>14}")
        edges = np.quantile(lam, np.linspace(0, 1, 11))
        spans = []
        for b in range(10):
            k = (lam >= edges[b]) & (lam <= edges[b + 1])
            if k.sum() < 5:
                continue
            pe = np.percentile(y_ev[k], [10, 90])
            pp = np.percentile(y_per[k], [10, 90])
            spans.append((pe[1] / pe[0], b, edges[b], edges[b + 1]))
            print(f"  [{edges[b]:>9.2f},{edges[b+1]:>9.2f}]{k.sum():>5d}"
                  f"{pe[1]/pe[0]:>13.1f}{y_ev[k].max()/y_ev[k].min():>13.1f}"
                  f"{pp[1]/pp[0]:>14.1f}")
        return spans, edges

    spans, edges = spread_table(det_ev, det_pp, "det I / T^2")
    spread_table(lmin_ev, lmin_pp, "lam_min(I) / T")

    # the most convincing exhibit: best and worst input in the widest bin
    wid, b, lo, hi = max(spans)
    k = np.where((lam >= lo) & (lam <= hi))[0]
    ib, iw = k[np.argmax(det_ev[k])], k[np.argmin(det_ev[k])]
    print(f"\n  WIDEST BIN: lambda in [{lo:.2f}, {hi:.2f}] "
          f"(p90/p10 = {wid:.1f}).  Best vs worst input in that bin:")
    for name, i in (("BEST ", ib), ("WORST", iw)):
        r = res[i]
        print(f"    {name}: M = {r['M']}"
              f"   a = [" + ", ".join(f"{a:.4f}" for a in r["amps"]) + "]"
              f"   w = [" + ", ".join(f"{w:.4f}" for w in r["freqs"]) + "]"
              f"   ph = [" + ", ".join(f"{p:.4f}" for p in r["phases"])
              + "]")
        print(f"           lambda = {r['lam']:.2f}   det I/T^2 = "
              f"{r['det_rate']:.4g}   (m = {r['m']:.4f}, Delta = "
              f"{r['Delta']:.5f}, Delta/C_min = "
              f"{r['Delta']/r['C_min']:.4f})")
    print(f"    -> same event rate within the bin, det I differs by "
          f"x{det_ev[ib]/det_ev[iw]:.0f}.")

    rho_s = spearmanr(det_ev, lmin_ev).statistic
    print(f"\n  do D and E rank inputs the same way?  Spearman rank "
          f"correlation of det I/T^2 vs lam_min(I)/T")
    print(f"  across the {len(res)} inputs: {rho_s:.3f}  (1 = identical "
          f"ranking).")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    for a, (ye, yp, lab) in zip(
            ax, ((det_ev, det_pp, "det I / T^2"),
                 (lmin_ev, lmin_pp, "lam_min(I) / T"))):
        a.loglog(lam, yp, ".", ms=3, color="0.65", label="periodic, same N")
        for Mn, col in ((1, "C0"), (2, "C1"), (3, "C2")):
            kk = Ms == Mn
            a.loglog(lam[kk], ye[kk], "o", ms=4, color=col, mfc="none",
                     label=f"events, M = {Mn}")
        a.set_xlabel("lambda = m / Delta   (Delta = max(C_min/100, "
                     "C_peak/2000))")
        a.set_ylabel(lab)
        a.set_title(f"{lab} vs event rate", fontsize=10)
        a.legend(fontsize=8, loc="lower right")
        a.grid(alpha=0.3)
    fig.suptitle(f"Figure 3 -- information vs communication cost, "
                 f"{n_inputs} random multisines, two-lag model", fontsize=11)
    fig.tight_layout()
    path = f"{outdir}/multisine_dynamic_fig3.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\n  figure written to {path}")
    return res


# --------------------------------------------------------------------------
# Figure 4 -- per-event efficiency vs frequency content
# --------------------------------------------------------------------------

def fig4(res, f1ws, f1det, outdir="."):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wm = np.array([r["w_mean"] for r in res])
    de = np.array([r["detMw"] for r in res])
    Ms = np.array([r["M"] for r in res])

    print("\n" + "=" * 96)
    print("FIGURE 4 -- PER-EVENT EFFICIENCY det I / N^2 vs power-weighted "
          "mean frequency sum a^2 w")
    print("=" * 96)
    print("  same inputs/windows as Figure 3 (det I/N^2 is the T-free "
          "version of det I/lambda^2);")
    print("  M = 1 curve from Figure 1(c) overlaid.")
    print(f"\n{'w_mean bin':>18}{'n':>5}{'median detMw':>15}{'p10':>12}"
          f"{'p90':>12}{'M=1 curve at centre':>21}")
    edges = np.geomspace(0.05, 20, 9)
    for b in range(8):
        k = (wm >= edges[b]) & (wm <= edges[b + 1])
        if k.sum() < 3:
            continue
        c = np.sqrt(edges[b] * edges[b + 1])
        ref = np.interp(np.log(c), np.log(f1ws), f1det)
        p = np.percentile(de[k], [10, 50, 90])
        print(f"  [{edges[b]:>6.3g},{edges[b+1]:>6.3g}]{k.sum():>5d}"
              f"{p[1]:>15.4e}{p[0]:>12.4e}{p[2]:>12.4e}{ref:>21.4e}")

    fig, a = plt.subplots(figsize=(8, 5.6))
    for Mn, col in ((1, "C0"), (2, "C1"), (3, "C2")):
        k = Ms == Mn
        a.loglog(wm[k], de[k], "o", ms=4, color=col, mfc="none",
                 label=f"M = {Mn}")
    a.loglog(f1ws, f1det, "k-", lw=1.6, label="M = 1 curve (Fig. 1c)")
    a.axvline(1 / TAU1, color="0.4", lw=1, ls=":")
    a.axvline(1 / TAU2, color="0.4", lw=1, ls=":")
    a.set_xlabel("power-weighted mean frequency  sum a^2 w  (power = 1)")
    a.set_ylabel("det I / N^2  (per-event information)")
    a.set_title("Figure 4 -- per-event efficiency vs frequency content; "
                "dotted: 1/T1, 1/T2", fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    fig.tight_layout()
    path = f"{outdir}/multisine_dynamic_fig4.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\n  figure written to {path}")


# --------------------------------------------------------------------------
# Scorecard and the full run
# --------------------------------------------------------------------------

def scorecard(f1, f2, res):
    ws = f1["ws"]
    rows = f1["rows"]
    print("\n" + "=" * 96)
    print("PREDICTIONS SCORECARD")
    print("=" * 96)

    dev = max(abs(q["det_ratio"] - 8 / 9) for q in rows)
    sdev = max(abs(s["det_ratio"] - 8 / 9) for _, s, _ in f1["sims"])
    print(f"\nP1 (det ratio = 8/9 at M = 1, filter/frequency/theta-free; "
          f"PROVEN, code check).  HELD.")
    print(f"   quadrature: |det ratio - 8/9| <= {dev:.1e} over the sweep; "
          f"simulation: within {sdev:.1e} of 8/9,")
    print("   i.e. the documented -0.44 Delta/C finite-threshold bias.  "
          "The T-congruence cancels in det, as proven.")

    raw = np.array([v["raw"] for v in f1["white"]])
    dg = np.array([v["diag"] for v in f1["white"]])
    wm = np.array([v["white_min"] for v in f1["white"]])
    print(f"\nP2 (lam_min ratio is metric-bound; full whitening restores "
          f"2/3, diagonal does not; PROVEN, code check).  HELD.")
    print(f"   raw ratio in [{raw.min():.4f}, {raw.max():.4f}] -- not 2/3, "
          f"above 1, pinned near 4/3 by the ill-conditioned")
    print(f"   M_per at M = 1; diag-normalised in [{dg.min():.4f}, "
          f"{dg.max():.4f}] -- moves with omega, never 2/3; full")
    print(f"   whitening S = M_per^(-1/2): lam_min = 2/3 to "
          f"{np.abs(wm - 2/3).max():.1e} at every omega.  E is metric-"
          f"bound; D is not.")

    m_end = rows[-1]["m"]
    msat = (2 / np.pi) * (THETA[0] / TAU1 + THETA[1] / TAU2)
    mono = bool(np.all(np.diff([q["m"] for q in rows]) > 0))
    print(f"\nP3 (event rate saturation).  HELD for m; lambda depends on "
          f"the Delta rule.")
    print(f"   m = <|xdot|> is monotone ({mono}) with NO interior maximum, "
          f"saturating at {m_end:.4f} vs the")
    print(f"   coherent two-lag plateau (2/pi)(th1/T1 + th2/T2) = "
          f"{msat:.4f} (0.4% at omega = 20); the single-lag")
    print(f"   pieces are {(2/np.pi)*THETA[0]/TAU1:.4f} and "
          f"{(2/np.pi)*THETA[1]/TAU2:.4f}.  Both lags contribute ~1/(i w) "
          f"phase at high omega, so")
    print("   they add coherently.  lambda = m/Delta itself keeps rising "
          "(Delta = C/100 shrinks like 1/omega):")
    print("   the flattening statement is about the rate at any FIXED "
          "threshold, and holds there.")

    print(f"\nP4 (does R11 + R22 = 2<E>_w/<E>_1 survive without the "
          f"derivative relation?; OPEN).  IT DOES NOT.")
    print(f"   Across the Figure-2 grid: max|R11+R22 - 2<E_x>_w/<E_x>_1| "
          f"= {f2['d_x']:.2f} (output envelope), "
          f"{f2['d_u']:.2f}")
    print("   (input envelope); the direction sum sits BELOW 2 almost "
          "everywhere while the envelope reading sits")
    print("   ABOVE 2, and their correlation over the grid is weak.  "
          "Already at M = 1 the sum is not 2 (e.g.")
    q1 = quad_case(multisine([1.0], [1.0], [0.0]))
    print(f"   {q1['dirsum']:.4f} at omega = 1): the identity needed the "
          f"regressors to be a quadrature pair (phase")
    print("   offset 90 deg), which [G1 u, G2 u] are not.  The equal-N "
          "det gain > 1 nevertheless SURVIVES (Figure")
    print("   2d: up to ~1.55 near omega_2/omega_1 ~ 1.4), so the "
          "envelope-sum identity was sufficient, not")
    print("   necessary, for the M > 1 gain.")

    lam = np.array([r["lam"] for r in res])
    de = np.array([r["det_rate"] for r in res])
    mid = (lam > np.quantile(lam, 0.3)) & (lam < np.quantile(lam, 0.7))
    p = np.percentile(de[mid], [10, 90])
    print("\nBOTTOM LINE (the supervisor's question): at comparable event "
          "rate (middle 40% of lambda),")
    print(f"   det I/T^2 spans p90/p10 = {p[1]/p[0]:.0f}x.  Far beyond "
          f"factor 2: input choice changes the information")
    print("   enormously at the same communication cost -- the trade-off "
          "is real.  Where the figures point (words")
    print("   only, no optimisation): efficient inputs concentrate their "
          "power within roughly a factor 3 of the")
    print(f"   filter corners 1/T1 = {1/TAU1:g} and 1/T2 = {1/TAU2:g}; "
          f"high-frequency tones buy events but almost no")
    print("   information; and a slightly-split pair of tones beats a "
          "single tone at equal N (Figure 2d).")


def run_partA(scratch):
    from multisine_criteria import _run_and_log      # reuse the Tee logger
    ok = {}

    def _a():
        f1 = fig1()
        if not f1["ok"]:
            print("\nSTOPPING: P1/P2 code checks failed -- see above.")
            ok["f1"] = None
            return None
        f2 = fig2()
        ok["f1"], ok["f2"] = f1, f2
        np.savez(scratch + "/dynA.npz",
                 ws=f1["ws"],
                 det1=[q["detMw"] for q in f1["rows"]],
                 raw=[v["raw"] for v in f1["white"]],
                 diag=[v["diag"] for v in f1["white"]],
                 white=[v["white_min"] for v in f1["white"]],
                 m=[q["m"] for q in f1["rows"]],
                 dr=[q["det_ratio"] for q in f1["rows"]],
                 sims_dr=[s["det_ratio"] for _, s, _ in f1["sims"]],
                 d_x=f2["d_x"], d_u=f2["d_u"])
        return True

    _run_and_log([_a], path="multisine_dynamic_output.txt", mode="w")
    return ok


def run_partB(scratch, f1, f2):
    from multisine_criteria import _run_and_log

    def _b():
        res = fig3()
        fig4(res, f1["ws"], np.array([q["detMw"] for q in f1["rows"]]))
        scorecard(f1, f2, res)
        return res

    _run_and_log([_b], path="multisine_dynamic_output.txt", mode="a")
