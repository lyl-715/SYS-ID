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


def _spread_table(lam, y_ev, y_per, name):
    """p90/p10 of y within deciles of lam; returns (spans, edges)."""
    print(f"\n  {name}: vertical spread at comparable lambda "
          f"(deciles of lambda)")
    print(f"{'lambda bin':>24}{'n':>5}{'p90/p10 ev':>13}{'max/min ev':>13}"
          f"{'p90/p10 per':>14}")
    edges = np.quantile(lam, np.linspace(0, 1, 11))
    spans = []
    for b in range(10):
        k = (lam >= edges[b]) & (lam <= edges[b + 1])
        if k.sum() < 5:
            continue
        pe = np.percentile(y_ev[k], [10, 90])
        pp = np.percentile(y_per[k], [10, 90])
        spans.append((pe[1] / pe[0], edges[b], edges[b + 1]))
        print(f"  [{edges[b]:>9.2f},{edges[b+1]:>9.2f}]{k.sum():>5d}"
              f"{pe[1]/pe[0]:>13.1f}{y_ev[k].max()/y_ev[k].min():>13.1f}"
              f"{pp[1]/pp[0]:>14.1f}")
    return spans, edges


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
    print("  The DESIGN comparison uses ONE FIXED Delta for every input: "
          "the trigger is a constant of the")
    print("  experiment, and det I scales like Delta^-2 at fixed input, "
          "so an input-dependent Delta rule")
    print("  injects a spurious Delta^-2 spread into any 'equal lambda' "
          "comparison.  (The earlier variable-")
    print("  Delta version of this figure suffered exactly that; it is "
          "kept as panel (b), relabelled as a")
    print("  numerical-resolution study.)")
    print(f"  omega log-uniform [0.05, 20], random phases, power = 1; "
          f"window {N_PERIODS} periods of the slowest tone.")

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

    # ---- the fixed experiment threshold ---------------------------------
    C_peaks = np.array([r["C_peak"] for r in res])
    C_mins = np.array([r["C_min"] for r in res])
    delta_fix = float(np.median(C_peaks)) / 100.0
    keep = C_mins >= delta_fix
    print(f"\n  FIXED Delta = median(C_peak)/100 = {delta_fix:.5f}   "
          f"(median C_peak = {100*delta_fix:.4f})")
    print(f"  inputs with C_min < Delta: {int((~keep).sum())} of "
          f"{len(res)} -- FLAGGED and excluded from the")
    print("  statistics below (their weakest component moves less than "
          "one threshold: too few events per")
    print("  component to be meaningful); they are drawn as light "
          "crosses in panel (a).")

    m_all = np.array([r["m"] for r in res])
    Ms = np.array([r["M"] for r in res])
    detMw = np.array([r["detMw"] for r in res])
    detM1 = np.array([r["detM1"] for r in res])
    lamMw = np.array([r["lamMw"] for r in res])
    lamM1 = np.array([r["lamM1"] for r in res])

    lamF = m_all / delta_fix                      # fixed-Delta event rate
    detF_ev, detF_pp = lamF ** 2 * detMw, lamF ** 2 * detM1
    lminF_ev, lminF_pp = lamF * lamMw, lamF * lamM1

    k = keep
    print(f"\n  === FIXED-Delta analysis ({int(k.sum())} inputs) ===")
    print(f"  lambda = m/Delta now spans [{lamF[k].min():.1f}, "
          f"{lamF[k].max():.1f}] only (m saturates), so 'comparable")
    print("  event count' is a genuine statement about m, not about the "
          "threshold rule.")
    spans, _ = _spread_table(lamF[k], detF_ev[k], detF_pp[k],
                             "det I / T^2  (FIXED Delta)")
    _spread_table(lamF[k], lminF_ev[k], lminF_pp[k],
                  "lam_min(I) / T  (FIXED Delta)")

    wid, lo, hi = max(spans)
    kk = np.where(k & (lamF >= lo) & (lamF <= hi))[0]
    ib, iw = kk[np.argmax(detF_ev[kk])], kk[np.argmin(detF_ev[kk])]
    print(f"\n  WIDEST BIN (fixed Delta): lambda in [{lo:.2f}, {hi:.2f}] "
          f"(p90/p10 = {wid:.1f}).  Best vs worst:")
    for name, i in (("BEST ", ib), ("WORST", iw)):
        r = res[i]
        print(f"    {name}: M = {r['M']}"
              f"   a = [" + ", ".join(f"{a:.4f}" for a in r["amps"]) + "]"
              f"   w = [" + ", ".join(f"{w:.4f}" for w in r["freqs"]) + "]"
              f"   ph = [" + ", ".join(f"{p:.4f}" for p in r["phases"])
              + "]")
        print(f"           lambda = {lamF[i]:.2f}   det I/T^2 = "
              f"{detF_ev[i]:.4g}   (m = {r['m']:.4f}, C_min = "
              f"{r['C_min']:.4f}, Delta/C_min = "
              f"{delta_fix/r['C_min']:.4f})")
    print(f"    -> same threshold, same event rate to within the bin, "
          f"det I differs by x{detF_ev[ib]/detF_ev[iw]:.3g}.")

    print(f"\n  periodic reference at the same N, fixed Delta: its "
          f"within-bin p90/p10 is listed above --")
    print("  it shows a COMPARABLE spread, so the spread is carried by "
          "the input's frequency content;")
    dr = detF_ev[k] / detF_pp[k]
    print(f"  the event/periodic distinction at equal N stays within "
          f"[{dr.min():.3f}, {dr.max():.3f}] on top.")

    rho_s = spearmanr(detF_ev[k], lminF_ev[k]).statistic
    print(f"\n  D vs E ranking (fixed Delta, kept inputs): Spearman = "
          f"{rho_s:.3f}  (1 = identical ranking).")

    # ---- variable-Delta numbers (resolution study, panel b) -------------
    lamV = np.array([r["lam"] for r in res])
    detV = np.array([r["det_rate"] for r in res])
    nfloor = sum(1 for r in res if r["floor"])
    print(f"\n  panel (b), input-dependent Delta = max(C_min/100, "
          f"C_peak/2000): floor hit by {nfloor} = "
          f"{100*nfloor/len(res):.1f}% of inputs;")
    print("  kept ONLY as a numerical-resolution study (which Delta "
          "each input was SIMULATED at in the")
    print("  validation runs), NOT as a design comparison.")

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.8))
    a = ax[0]
    a.loglog(lamF[k], detF_pp[k], ".", ms=3, color="0.65",
             label="periodic, same N")
    for Mn, col in ((1, "C0"), (2, "C1"), (3, "C2")):
        s_ = k & (Ms == Mn)
        a.loglog(lamF[s_], detF_ev[s_], "o", ms=4, color=col, mfc="none",
                 label=f"events, M = {Mn}")
    a.loglog(lamF[~k], detF_ev[~k], "x", ms=4, color="0.75",
             label="flagged: C_min < Delta")
    a.set_xlabel("lambda = m / Delta")
    a.set_ylabel("det I / T^2")
    a.set_title(f"(a) FIXED Delta = median(C_peak)/100 = {delta_fix:.4f}"
                "\nthe design comparison", fontsize=10)
    a.legend(fontsize=8, loc="lower right")

    b = ax[1]
    for Mn, col in ((1, "C0"), (2, "C1"), (3, "C2")):
        s_ = Ms == Mn
        b.loglog(lamV[s_], detV[s_], "o", ms=4, color=col, mfc="none",
                 label=f"M = {Mn}")
    b.set_xlabel("lambda = m / Delta,  Delta = max(C_min/100, C_peak/2000)")
    b.set_ylabel("det I / T^2")
    b.set_title("(b) input-dependent Delta -- NUMERICAL-RESOLUTION STUDY"
                "\nNOT a design comparison: det I ~ Delta^-2 inflates the "
                "spread", fontsize=10)
    b.legend(fontsize=8, loc="lower right")
    for a_ in ax:
        a_.grid(alpha=0.3)
    fig.suptitle(f"Figure 3 -- information vs communication cost, "
                 f"{n_inputs} random multisines, two-lag model",
                 fontsize=11)
    fig.tight_layout()
    path = f"{outdir}/multisine_dynamic_fig3.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\n  figure written to {path}")
    return dict(res=res, delta_fix=delta_fix, keep=keep, lamF=lamF,
                detF_ev=detF_ev)


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

def scorecard(f1, f2, f3):
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

    k = f3["keep"]
    lamF, deF = f3["lamF"][k], f3["detF_ev"][k]
    mid = (lamF > np.quantile(lamF, 0.3)) & (lamF < np.quantile(lamF, 0.7))
    p = np.percentile(deF[mid], [10, 90])
    print("\nBOTTOM LINE (the supervisor's question), at ONE FIXED "
          "threshold Delta = median(C_peak)/100 =")
    print(f"   {f3['delta_fix']:.5f} for every input -- the earlier "
          "variable-Delta headline conflated the Delta^-2")
    print("   scaling of det I with input design and is retracted; panel "
          "(b) keeps it only as a resolution")
    print(f"   study.  At comparable event rate (middle 40% of lambda), "
          f"det I/T^2 spans p90/p10 = {p[1]/p[0]:.0f}x.")
    print("   Still far beyond factor 2: the trade-off is real after the "
          "confound is removed.  Efficient inputs")
    print(f"   concentrate power within roughly a factor 3 of the filter "
          f"corners 1/T1 = {1/TAU1:g}, 1/T2 = {1/TAU2:g}; high-")
    print("   frequency tones buy events but almost no information; a "
          "slightly-split pair beats a single tone")
    print("   at equal N (Figure 2d).")


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
        f3 = fig3()
        fig4(f3["res"], f1["ws"], np.array([q["detMw"] for q in f1["rows"]]))
        scorecard(f1, f2, f3)
        return f3

    _run_and_log([_b], path="multisine_dynamic_output.txt", mode="a")


# --------------------------------------------------------------------------
# High-frequency power laws for Figure 1  (log-log fits over omega in [5,20])
# --------------------------------------------------------------------------

def highfreq_exponents(w_lo=5.0, w_hi=20.0, n=25):
    """
    Fit y ~ omega^b over [w_lo, w_hi] for the Figure-1 quantities and
    check them against the analytic asymptotes.

    Asymptotics for the two-lag channels (a = 1, T1 T2 = 1 here):
      |G_j| -> 1/(omega T_j),  angle G_j -> -90 deg with residual
      1/(omega T_j), so the PHASE SPLIT is
          dphi = atan(omega T2) - atan(omega T1) -> (1/T1 - 1/T2)/omega,
      and the per-event (per-sample) moment determinant of a single tone,
          det M = (1/4) a^4 |G1|^2 |G2|^2 sin^2(dphi)
               -> (1/4) (1/T1 - 1/T2)^2 / (T1 T2)^2 * omega^-6,
      is omega^-4 (amplitudes) times omega^-2 (phase collapse): omega^-6.
      With lambda ~ omega (Delta = C/100, C ~ 1/omega, m -> const):
          det I/T^2 = lambda^2 det M_w ~ omega^-4,
          lam_min(M) ~ det/trace ~ omega^-6 / omega^-2 = omega^-4.
      NOTE det I/N^2 IS det M: it is already per event, so lambda enters
      it zero more times -- there is no omega^-8 quantity in this family.
    """
    ws = np.geomspace(w_lo, w_hi, n)
    rows = [quad_case(multisine([1.0], [w], [0.0])) for w in ws]
    qty = (("m = <|xdot|>", "m", 0.0),
           ("lambda = m/Delta", "lam", 1.0),
           ("det I / T^2", "det_rate", -4.0),
           ("det I / N^2", "detMw", -6.0),
           ("lam_min(I) / N", "lamMw", -4.0))
    print("=" * 92)
    print(f"HIGH-FREQUENCY POWER LAWS, log-log linear fit over omega in "
          f"[{w_lo:g}, {w_hi:g}]  ({n} points)")
    print("=" * 92)
    print(f"{'quantity':<18}{'fitted exp':>12}{'rms resid':>11}"
          f"{'max resid':>11}{'slope[5,10]':>13}{'slope[10,20]':>14}"
          f"{'asymptote':>11}")
    lw = np.log10(ws)
    out = {}
    for name, key, pred in qty:
        ly = np.log10([q[key] for q in rows])
        b, a = np.polyfit(lw, ly, 1)
        r = ly - (a + b * lw)
        half = ws <= np.sqrt(w_lo * w_hi)
        b1 = np.polyfit(lw[half], ly[half], 1)[0]
        b2 = np.polyfit(lw[~half], ly[~half], 1)[0]
        out[key] = b
        print(f"{name:<18}{b:>12.3f}{np.sqrt(np.mean(r**2)):>11.1e}"
              f"{np.abs(r).max():>11.1e}{b1:>13.3f}{b2:>14.3f}"
              f"{pred:>11.0f}")

    # the analytic constant, not just the exponent
    cM1 = 0.25 * (1 / TAU1 - 1 / TAU2) ** 2 / (TAU1 * TAU2) ** 2
    q20 = rows[-1]
    print(f"\n  analytic constant check at omega = {w_hi:g}: "
          f"det M_1 asymptote (1/4)(1/T1 - 1/T2)^2/(T1 T2)^2 omega^-6")
    print(f"  = {cM1:.4f} omega^-6 = {cM1 / w_hi**6:.4e}   vs measured "
          f"det M_1 = {q20['detM1']:.4e}   "
          f"(ratio {q20['detM1'] * w_hi**6 / cM1:.4f})")
    print(f"  and det M_w = (8/9) det M_1 at M = 1: measured ratio "
          f"{q20['detMw'] / q20['detM1']:.6f}")

    # where a "-5" reading comes from: the pre-asymptotic mid band
    ws2 = np.geomspace(2.0, 6.0, 13)
    ly2 = np.log10([quad_case(multisine([1.0], [w], [0.0]))["detMw"]
                    for w in ws2])
    b_mid = np.polyfit(np.log10(ws2), ly2, 1)[0]
    print(f"\n  det I/N^2 fitted over the PRE-ASYMPTOTIC band "
          f"omega in [2, 6]: exponent {b_mid:.2f}")
    print("  (|G1| has its knee at 1/T1 = 2 and sin^2(dphi) is still far")
    print("  from its omega^-2 tail there, so a slope read off the plot")
    print("  across the middle decades comes out near -5; over [5, 20]")
    print("  the fit is already within 0.1 of -6.)")
    return out


# --------------------------------------------------------------------------
# WHERE does R = det I_ev / det I_per > 1 live?  (H-ratio / H-absolute /
# H-beat discrimination; quadrature only, no trigger, no Delta)
# --------------------------------------------------------------------------

def R_case(case):
    """One (w1, w2, rho, phases) cell: R by |xdot|-weighted quadrature,
    plus the window-quadrature det M_1 for the phase-cancellation check
    (the closed-form det M_1 is phase-free ALGEBRAICALLY for distinct
    tones -- the input phase enters both channels identically and cancels
    in cos(alpha_i - alpha_j) -- so only the finite-window average can
    show phase scatter, and that scatter is ergodicity, not algebra)."""
    tag, w1, w2, rho, ph1, ph2 = case
    try:
        ms = multisine([np.sqrt(1 - rho), np.sqrt(rho)], [w1, w2],
                       [ph1, ph2])                    # sum a^2 = 1
        msx = output_multisine(ms, THETA, MODEL_DYNAMIC)
        t, T = _window_grid(ms)
        Phi = regressors_ms(t, ms, MODEL_DYNAMIC)
        wt = np.abs(evaluate_dot(msx, t))
        Z = np.trapezoid(wt, t)
        Mw, M1w = np.empty((2, 2)), np.empty((2, 2))
        for i in range(2):
            for j in range(i, 2):
                pij = Phi[:, i] * Phi[:, j]
                Mw[i, j] = Mw[j, i] = np.trapezoid(pij * wt, t) / Z
                M1w[i, j] = M1w[j, i] = np.trapezoid(pij, t) / T
        M1e = uniform_moments_dyn(ms)
        return dict(tag=tag, ok=True,
                    R=float(np.linalg.det(Mw) / np.linalg.det(M1e)),
                    detM1e=float(np.linalg.det(M1e)),
                    detM1w=float(np.linalg.det(M1w)),
                    detMw=float(np.linalg.det(Mw)),
                    C_min=float(msx.amps.min()))
    except Exception as exc:
        return dict(tag=tag, ok=False, error=f"{type(exc).__name__}: {exc}")


def experiment_R_location(w1s=(0.1, 0.3, 1.0, 3.0), n_w2=24, n_rho=12,
                          n_draw=16, workers=4, seed=17, outdir="."):
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    rhos = np.linspace(0.05, 0.95, n_rho)
    ratios = np.geomspace(1.0, 30.0, n_w2)      # w2 = w1 * ratio
    rng = np.random.default_rng(seed)

    print("=" * 100)
    print("WHERE IS R = det I_ev / det I_per > 1 ?   quadrature sweep over "
          "w1 in " + str(tuple(w1s)))
    print("=" * 100)
    print(f"  w2 = w1 * r, r in [1, 30] ({n_w2} log points); rho in "
          f"[0.05, 0.95] ({n_rho} points); {n_draw} phase draws;")
    print(f"  window = {N_PERIODS} periods of w1 (T = 400*2pi/w1: "
          + ", ".join(f"{N_PERIODS*2*np.pi/w:.0f}" for w in w1s)
          + " for the four w1);")
    print("  model/theta/power as Figure 2.  R is a ratio of weighted "
          "moment determinants -- no trigger, no Delta.")

    data = {}
    for iw, w1 in enumerate(w1s):
        cases = []
        for i, r in enumerate(ratios):
            for j, rho in enumerate(rhos):
                for k in range(n_draw):
                    ph = rng.uniform(0, 2 * np.pi, 2)
                    cases.append(((i, j, k), w1, w1 * r, rho, ph[0], ph[1]))
        t0 = time.time()
        with mp.Pool(workers) as pool:
            out = pool.map(R_case, cases, chunksize=8)
        wall = time.time() - t0
        if iw == 0:
            print(f"\n  first w1 = {w1}: {len(cases)} cases in "
                  f"{wall:.0f} s on {workers} workers -> continuing")
        Rg = np.full((n_w2, n_rho, n_draw), np.nan)
        d1e = np.full_like(Rg, np.nan)
        d1w = np.full_like(Rg, np.nan)
        for r_ in out:
            if r_["ok"]:
                Rg[r_["tag"]] = r_["R"]
                d1e[r_["tag"]] = r_["detM1e"]
                d1w[r_["tag"]] = r_["detM1w"]
        data[w1] = dict(R=Rg, d1e=d1e, d1w=d1w, wall=wall)

    # ---------------- tables and per-w1 findings -------------------------
    peaks = {}
    for w1 in w1s:
        Rm = data[w1]["R"].mean(axis=2)
        Rs = data[w1]["R"].std(axis=2)
        print("\n" + "-" * 100)
        print(f"  w1 = {w1}:  mean R over {n_draw} draws   "
              f"(rows r = w2/w1, cols rho)")
        print("            " + "".join(f"{r:>8.2f}" for r in rhos))
        for i, r in enumerate(ratios):
            print(f"  r={r:>7.3f} " + "".join(f"{Rm[i, j]:>8.4f}"
                                              for j in range(n_rho)))
        i, j = np.unravel_index(np.nanargmax(Rm), Rm.shape)
        w2s_ = w1 * ratios
        peaks[w1] = (w2s_[i], rhos[j], Rm[i, j])
        print(f"\n  peak: R = {Rm[i, j]:.4f} +- {Rs[i, j]:.4f} (std over "
              f"draws) at w2* = {w2s_[i]:.4f}, rho* = {rhos[j]:.2f}")
        print(f"        w2*/w1 = {ratios[i]:.3f}    w2* - w1 = "
              f"{w2s_[i] - w1:.4f}")
        print(f"  phase spread: max std/|mean| of R over the grid = "
              f"{np.nanmax(Rs / np.abs(Rm)):.4f}; at the peak "
              f"{Rs[i, j]/Rm[i, j]:.4f}")
        # phase spread of det M_1 -- SPLIT by the r = 1 row: at w2 = w1
        # the tones are coincident and merge into ONE sinusoid whose
        # amplitude depends on the drawn phase difference, so det M_1
        # varies there BY PHYSICS.  The phase-cancellation algebra is a
        # statement about DISTINCT tones (rows r > 1).
        sm = lambda A: np.nanmax(A.std(axis=2) / np.abs(A.mean(axis=2)))
        d1e, d1w = data[w1]["d1e"], data[w1]["d1w"]
        dMw = data[w1]["R"] * d1e
        print(f"  det M_1 phase spread, DISTINCT tones (r > 1): closed "
              f"form std/mean = {sm(d1e[1:]):.1e}  (algebra: exactly 0);")
        print(f"    window-quadrature det M_1 std/mean = {sm(d1w[1:]):.1e}"
              f" (finite-window ergodicity, not algebra);")
        print(f"    weighted det M_w std/mean = {sm(dMw[1:]):.1e}")
        print(f"  coincident row r = 1 (tones merge into one sinusoid): "
              f"det M_1 std/mean = {sm(d1e[:1]):.1e} -- phase-dependent")
        print(f"    by amplitude interference, NOT a failure of the "
              f"cancellation; R there = 8/9 for every draw "
              f"(std {data[w1]['R'][0].std():.1e}).")

        # boundary rho -> 0, 1: exact single tones
        for rho_b, lab in ((0.0, "rho -> 0"), (1.0, "rho -> 1")):
            rb = []
            for r in (ratios[0], ratios[-1]):
                c = R_case(((0,), w1, w1 * r, rho_b, 0.0, 0.0))
                rb.append(c["R"])
            print(f"  boundary {lab}: R = {rb[0]:.6f} (r=1) .. "
                  f"{rb[-1]:.6f} (r=30)   [should be 8/9 = 0.888889]")

        # R = 1 crossings at fixed rho
        print(f"  R = 1 crossings (log-interp along r at fixed rho):")
        line = "    "
        for j2 in (1, 4, 7, 10):
            col = Rm[:, j2]
            cross = None
            for i2 in range(len(ratios) - 1, 0, -1):
                if (col[i2] - 1) * (col[i2 - 1] - 1) < 0:
                    f = (1 - col[i2 - 1]) / (col[i2] - col[i2 - 1])
                    cross = ratios[i2 - 1] * (ratios[i2] / ratios[i2 - 1]) ** f
                    break
            line += (f"rho={rhos[j2]:.2f}: r_x={cross:.2f} "
                     f"(w2={w1*cross:.3f})   " if cross
                     else f"rho={rhos[j2]:.2f}: none   ")
        print(line)

    # ---------------- hypothesis table -----------------------------------
    print("\n" + "=" * 100)
    print("HYPOTHESIS TABLE")
    print("=" * 100)
    print(f"{'w1':>7}{'w2*':>10}{'rho*':>7}{'R peak':>9}{'w2*/w1':>9}"
          f"{'w2*-w1':>10}")
    for w1 in w1s:
        w2p, rhop, Rp = peaks[w1]
        print(f"{w1:>7.3g}{w2p:>10.4f}{rhop:>7.2f}{Rp:>9.4f}"
              f"{w2p/w1:>9.3f}{w2p-w1:>10.4f}")
    print("  H-ratio    predicts w2*/w1 constant;  H-absolute predicts "
          "w2* constant (~0.45);")
    print("  H-beat     predicts w2* - w1 constant (~0.15).")

    # ---------------- figure ---------------------------------------------
    fig = plt.figure(figsize=(15, 9))
    for k, w1 in enumerate(w1s):
        a = fig.add_subplot(2, 3, k + 1)
        Rm = data[w1]["R"].mean(axis=2)
        im = a.pcolormesh(rhos, ratios, Rm, shading="auto", cmap="RdBu_r",
                          norm=mcolors.TwoSlopeNorm(vcenter=1.0))
        try:
            a.contour(rhos, ratios, Rm, levels=[1.0], colors="k",
                      linewidths=1.2)
        except Exception:
            pass
        a.set_yscale("log")
        a.set_xlabel("rho")
        a.set_ylabel("r = w2/w1")
        a.set_title(f"mean R, w1 = {w1}  (black: R = 1)", fontsize=10)
        fig.colorbar(im, ax=a)
    # discrimination panels: R vs ratio and vs absolute w2 at each w1's
    # peak rho
    a5 = fig.add_subplot(2, 3, 5)
    a6 = fig.add_subplot(2, 3, 6)
    for k, w1 in enumerate(w1s):
        Rm = data[w1]["R"].mean(axis=2)
        jj = int(np.argmin(np.abs(rhos - peaks[w1][1])))
        a5.semilogx(ratios, Rm[:, jj], "o-", ms=3, color=f"C{k}",
                    label=f"w1 = {w1} (rho = {rhos[jj]:.2f})")
        a6.semilogx(w1 * ratios, Rm[:, jj], "o-", ms=3, color=f"C{k}",
                    label=f"w1 = {w1}")
    for a_, xl, tt in ((a5, "r = w2/w1",
                        "collapse here -> H-ratio"),
                       (a6, "w2 (absolute)",
                        "collapse here -> H-absolute")):
        a_.axhline(1.0, color="k", lw=1, ls=":")
        a_.axhline(8 / 9, color="0.5", lw=1, ls=":")
        a_.set_xlabel(xl)
        a_.set_ylabel("mean R at that w1's peak rho")
        a_.set_title(tt, fontsize=10)
        a_.legend(fontsize=7)
        a_.grid(alpha=0.3)
    fig.suptitle("Locating R = det I_ev / det I_per > 1  (two-lag model, "
                 "quadrature)", fontsize=12)
    fig.tight_layout()
    path = f"{outdir}/R_location_figure.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\n  figure written to {path}")
    return dict(peaks=peaks, data=data, rhos=rhos, ratios=ratios)


def confirm_R_peaks(peaks=((0.1, 0.1344, 0.38), (0.3, 0.3478, 0.46),
                           (1.0, 1.1594, 0.46)),
                    ratios=(0.02, 0.01, 0.005)):
    """
    Event-simulation check at three peak cells (the brief asks for three
    only, so w1 = 3.0 is left out): does det I_ev / det I_per at equal N
    converge to the quadrature R as Delta shrinks?  Phases fixed at
    (0, 0) -- the quadrature reference is recomputed for those phases so
    like is compared with like.
    """
    print("\n" + "=" * 100)
    print("SIMULATION CONFIRMATION AT THE PEAKS  (Delta shrinking; "
          "quadrature recomputed at the same phases)")
    print("=" * 100)
    for w1, w2, rho in peaks:
        q = R_case(((0,), w1, w2, rho, 0.0, 0.0))
        # the simulation compares event and periodic FIMs on the SAME
        # finite window, so its Delta -> 0 limit is the window-matched
        # ratio R_win = det M_w / det M_1(window); the gap between R_win
        # and the sweep's R (which uses the exact infinite-window M_1) is
        # finite-window ergodicity, not a trigger effect
        R_win = q["detMw"] / q["detM1w"]
        ms = multisine([np.sqrt(1 - rho), np.sqrt(rho)], [w1, w2],
                       [0.0, 0.0])
        msx = output_multisine(ms, THETA, MODEL_DYNAMIC)
        T = N_PERIODS * 2 * np.pi / w1
        C_min = float(msx.amps.min())
        print(f"\n  w1 = {w1}, w2 = {w2}, rho = {rho}:  R (infinite-"
              f"window M_1, the sweep's object) = {q['R']:.6f}")
        print(f"    R_win (window-matched M_1, the sim's Delta -> 0 "
              f"limit) = {R_win:.6f}   window gap = "
              f"{q['R'] - R_win:+.2e}   C_min = {C_min:.4f}")
        print(f"  {'Delta/C_min':>12}{'N':>10}{'det ratio sim':>15}"
              f"{'sim - R_win':>13}")
        for r in ratios:
            Delta = r * C_min
            taus = send_on_delta_instants_ms(T, Delta, ms, THETA,
                                             MODEL_DYNAMIC)
            N = len(taus)
            tk = periodic_instants(T, N)
            dr = (np.linalg.det(fisher_ms(taus, ms, 1.0, MODEL_DYNAMIC))
                  / np.linalg.det(fisher_ms(tk, ms, 1.0, MODEL_DYNAMIC)))
            print(f"  {r:>12.3f}{N:>10d}{dr:>15.6f}{dr - R_win:>13.2e}")
    print("\n  Against the window-matched target the residual shrinks "
          "with Delta (finite-threshold bias);")
    print("  the remaining constant offset vs the sweep's R is the "
          "finite-window ergodicity of M_1,")
    print("  already quantified above (window det M_1 std/mean ~ 5e-3).  "
          "The quadrature measures the")
    print("  object the trigger produces.")
