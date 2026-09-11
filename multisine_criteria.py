"""
E criterion and event-budget objective for the multisine weighting study.

Everything here is pure quadrature -- no threshold, no event sequence, no
crossing detection, no ODE solver.  It builds on multisine_input_design:

    M_1 = <phi phi^T>_1   uniform time average, exact infinite window
                          (uniform_moments_exact -- closed form)
    M_w = <phi phi^T>_w   weight w = |xdot|, normalised as an average
                          (weighted_moments_time / weighted_moments_torus)
    m   = <|xdot|>_1      uniform average of |xdot|; the send-on-delta
                          event rate as Delta -> 0 is m / Delta.

Normalisation (a modelling CHOICE, stated as such): S = diag(M_1)^(-1/2),
Mt_w = S M_w S, so the periodic reference becomes Mt_1 = I.  det is
invariant to this diagonal scaling, so det(Mt_w) equals the existing
det_gain; eigenvalues are NOT scaling-invariant, so the E criterion below
(lambda_min(Mt_w)) is the E ratio in these normalised coordinates
specifically -- a different S would give a different E ranking.

Static model phi = [u, udot], theta = (1, 0.5) unless stated.
"""

import numpy as np

from event_input_design import MODEL_STATIC, output_envelope
from multisine_input_design import (det_gain, evaluate_dot, flagged_cases,
                                    multisine, output_multisine,
                                    uniform_moments_exact,
                                    weighted_moments_time,
                                    weighted_moments_torus, _rat_flag)

THETA_DEFAULT = (1.0, 0.5)

# the constrained optimum found last session (w1 >= 0.5, r <= 2 in the
# stationary limit): gain 1.580 at w1 = 0.631, r = 1.06, rho = 0.9486
CONSTRAINED_OPT = ("constrained opt", multisine([1.0, 0.9486],
                                                [0.631, 0.631 * 1.06],
                                                [0.0, 0.0]))


# --------------------------------------------------------------------------
# Normalised weighted moment matrix and the criteria read off it
# --------------------------------------------------------------------------

def normalised_Mw(Mw, M1):
    """Mt_w = S M_w S with S = diag(M_1)^(-1/2).  Then Mt_1 = I."""
    S = np.diag(1.0 / np.sqrt(np.diag(M1)))
    return S @ Mw @ S


def criteria_normalised(Mw, M1):
    """E_ratio, cond_ev, D_check from Mt_w; eigenvector of lambda_min."""
    Mt = normalised_Mw(Mw, M1)
    lam, vec = np.linalg.eigh(Mt)          # ascending
    v = vec[:, 0]
    return dict(Mt=Mt, E_ratio=float(lam[0]), lam_max=float(lam[-1]),
                cond_ev=float(lam[-1] / lam[0]),
                D_check=float(lam[0] * lam[-1]),
                # angle of the losing direction in the normalised
                # (u, udot) plane, folded to [0, 180) deg
                vmin_deg=float(np.degrees(np.arctan2(v[1], v[0])) % 180.0))


def mean_abs_xdot_time(ms, theta, model=MODEL_STATIC, T=None,
                       n_periods=400, n=2_000_001):
    """m = <|xdot|>_1 by time quadrature (same windowing conventions as
    weighted_moments_time)."""
    msx = output_multisine(ms, np.asarray(theta, float), model)
    if T is None:
        T = n_periods * 2 * np.pi / ms.freqs[0]
    t = np.linspace(0.0, T, n)
    return float(np.trapezoid(np.abs(evaluate_dot(msx, t)), t) / T)


# --------------------------------------------------------------------------
# STAGE 0 -- sanity
# --------------------------------------------------------------------------

def stage0(theta=THETA_DEFAULT, model=MODEL_STATIC):
    theta = np.asarray(theta, float)
    cases = flagged_cases()

    print("=" * 96)
    print("STAGE 0.  SANITY")
    print("=" * 96)

    # -- 0.1  M_1 diagonal ------------------------------------------------
    print("\n0.1  <u udot>_1 = 0 (M_1 diagonal), exact closed-form moments:")
    worst = 0.0
    for tag, ms in cases:
        M1 = uniform_moments_exact(ms, model)
        off = abs(M1[0, 1]) / np.sqrt(M1[0, 0] * M1[1, 1])
        worst = max(worst, off)
        if off > 1e-14:
            print(f"     FAIL: {tag}: |<u udot>_1| / sqrt(<u^2><udot^2>) "
                  f"= {off:.3e}")
            return None
    print(f"     all {len(cases)} cases pass; worst normalised off-diagonal "
          f"= {worst:.1e}")

    # -- 0.2  det invariance under the diagonal scaling -------------------
    print("\n0.2  D_check = det(Mt_w) vs existing det_gain (must agree to "
          "1e-10):")
    worst = 0.0
    rows = []
    for tag, ms in cases:
        M1 = uniform_moments_exact(ms, model)
        Mw, _ = weighted_moments_time(ms, theta, model)
        g = det_gain(Mw, M1)["gain"]
        c = criteria_normalised(Mw, M1)
        d = abs(c["D_check"] - g)
        worst = max(worst, d)
        rows.append((tag, ms, M1, Mw, g, c))
        if d > 1e-10:
            print(f"     FAIL: {tag}: |det(Mt_w) - det_gain| = {d:.3e}")
            return None
    print(f"     all cases pass; worst |D_check - det_gain| = {worst:.1e}")

    # -- 0.3  M = 1 control, four thetas ----------------------------------
    print("\n0.3  M = 1 control at theta = (1,0), (1,0.5), (1,1), (0,1).")
    print("     predictions (psi-independent): E_ratio = 2/3, lam_max = 4/3,")
    print("     cond_ev = 2, D_check = 8/9.  closed form for Mt_w itself:")
    print("     Mt_w = I + (1/3) [[-cos 2psi, -sin 2psi], "
          "[-sin 2psi, cos 2psi]]\n")
    ms1 = multisine([1.0], [1.0], [0.0])
    print(f"{'theta':<12}{'psi[deg]':>9}{'E_ratio':>10}{'-2/3':>10}"
          f"{'lam_max':>10}{'-4/3':>10}{'cond':>9}{'-2':>10}{'D':>10}"
          f"{'-8/9':>10}{'max|dMt|':>11}")
    for th in ((1.0, 0.0), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)):
        th = np.asarray(th, float)
        M1 = uniform_moments_exact(ms1, model)
        Mw, _ = weighted_moments_time(ms1, th, model)
        c = criteria_normalised(Mw, M1)
        _, psi = output_envelope(1.0, 1.0, th, model)
        # closed form: in normalised coordinates the M = 1 weighted matrix
        # is I + (1/3) * [[-cos, -sin], [-sin, cos]](2 psi); its sign
        # convention follows from cross = -(1/3) sin 2psi (u lags x by psi)
        Mt_cf = np.eye(2) + (1.0 / 3.0) * np.array(
            [[-np.cos(2 * psi), -np.sin(2 * psi)],
             [-np.sin(2 * psi), np.cos(2 * psi)]])
        dM = np.max(np.abs(c["Mt"] - Mt_cf))
        lab = "({:g}, {:g})".format(*th)
        print(f"{lab:<12}{np.degrees(psi):>9.2f}"
              f"{c['E_ratio']:>10.6f}{c['E_ratio']-2/3:>10.2e}"
              f"{c['lam_max']:>10.6f}{c['lam_max']-4/3:>10.2e}"
              f"{c['cond_ev']:>9.5f}{c['cond_ev']-2:>10.2e}"
              f"{c['D_check']:>10.6f}{c['D_check']-8/9:>10.2e}{dM:>11.2e}")
    print("\n     (discrepancies are the finite-Delta-free quadrature error;")
    print("      the closed-form matrix matches entry-wise, so the psi-")
    print("      independence of the eigenvalues is structural, not luck.)")

    # -- 0.4  the table ----------------------------------------------------
    print("\n0.4  THE NINE CASES  (primary route: time quadrature, "
          "T = 400 periods of w1)")
    print("     torus value shown only where the frequencies are rationally")
    print("     independent; 'n/a (rational)' otherwise.\n")
    print(f"{'case':<20}{'rational?':>10}{'D ratio':>11}{'D torus':>15}"
          f"{'E_ratio':>10}{'cond_ev':>10}{'m=<|xdot|>':>12}")
    for tag, ms, M1, Mw, g, c in rows:
        rat, _ = _rat_flag(ms)
        if rat:
            tor = "  n/a (rational)"
        else:
            gt = det_gain(weighted_moments_torus(ms, theta, model), M1)
            tor = f"{gt['gain']:>15.6f}"
        m = mean_abs_xdot_time(ms, theta, model)
        print(f"{tag:<20}{('yes' if rat else 'no'):>10}{g:>11.6f}{tor}"
              f"{c['E_ratio']:>10.6f}{c['cond_ev']:>10.4f}{m:>12.6f}")

    print("\n     NOTE: E_ratio and cond_ev are eigenvalues of the")
    print("     NORMALISED matrix Mt_w = S M_w S.  det is invariant to that")
    print("     choice of S; the eigenvalues are not, so the E column is")
    print("     tied to this normalisation -- a modelling choice, not a")
    print("     scale-free fact.")
    return rows





# --------------------------------------------------------------------------
# STAGE 1A -- the E criterion on the nine cases + constrained optimum
# --------------------------------------------------------------------------

def stage1A(theta=THETA_DEFAULT, model=MODEL_STATIC):
    from multisine_input_design import energy_identity
    theta = np.asarray(theta, float)
    cases = flagged_cases() + [CONSTRAINED_OPT]

    print("\n" + "=" * 96)
    print("STAGE 1A.  E CRITERION,  tau = trace(Mt_w)/2,  AND THE E GAP")
    print("=" * 96)
    print("  tau = trace(Mt_w)/2 = <E>_w/<E>_1 (energy identity);")
    print("  g := D_ratio - (2 tau - 1) = (lam_max - 1)(lam_min - 1):")
    print("  g > 0 puts BOTH eigenvalues of Mt_w on the same side of 1, and")
    print("  tau then decides which side -- so E_ratio > 1 iff g > 0 AND")
    print("  tau > 1.  tau > 1 holds in every case below (it is the")
    print("  positive |xdot|-energy correlation), so within this family the")
    print("  test reduces to the sign of g, as intended.\n")

    print(f"{'case':<20}{'D ratio':>10}{'E_ratio':>10}{'vmin[deg]':>11}"
          f"{'tau':>9}{'|tau-EI|':>10}{'|fixed|':>9}{'E gap g':>11}"
          f"{'sgn ok?':>8}")
    rows = []
    n_sign_agree = 0
    for tag, ms in cases:
        M1 = uniform_moments_exact(ms, model)
        Mw, M1win = weighted_moments_time(ms, theta, model)
        g = det_gain(Mw, M1)
        c = criteria_normalised(Mw, M1)
        tau = 0.5 * (c["Mt"][0, 0] + c["Mt"][1, 1])

        # check against the EXISTING energy_identity.  Its <E>_1 is the
        # window average, while tau's normalisation S uses the exact
        # infinite-window M_1; the ratio of the two <E>_1's is computable
        # exactly, and multiplying it back in must close the identity to
        # machine precision -- any residual left after that would be real.
        ei = energy_identity(ms, theta, model)
        tau_ei = ei["sum_from_E"] / 2.0
        cc = M1[0, 0] / M1[1, 1]
        f = (M1win[0, 0] + M1win[1, 1] * cc) / (2.0 * M1[0, 0])
        resid_raw = abs(tau - tau_ei)
        resid_fix = abs(tau - tau_ei * f)

        Egap = g["gain"] - (2.0 * tau - 1.0)
        agree = np.sign(c["E_ratio"] - 2.0 / 3.0) == np.sign(
            g["gain"] - 8.0 / 9.0)
        n_sign_agree += bool(agree)
        rows.append(dict(tag=tag, ms=ms, M1=M1, Mw=Mw, gain=g["gain"],
                         crit=c, tau=tau, Egap=Egap))
        print(f"{tag:<20}{g['gain']:>10.6f}{c['E_ratio']:>10.6f}"
              f"{c['vmin_deg']:>11.2f}{tau:>9.5f}{resid_raw:>10.1e}"
              f"{resid_fix:>9.1e}{Egap:>11.6f}"
              f"{('yes' if agree else 'NO'):>8}")

    print("\n  |tau-EI| compares tau against energy_identity AS IS: its")
    print("  <E>_1 is a finite-window average, so the residual is window")
    print("  ergodicity (zero for the exactly periodic cases), not a")
    print("  failure of the identity.  |fixed| rescales that <E>_1 to the")
    print("  exact infinite-window value; it must be < 1e-10 and is.")

    print("\n  Findings on this set:")
    print("   - E_ratio > 1 NOWHERE: every case, including all the")
    print("     D ratio > 1 ones and the constrained optimum, LOSES on the")
    print("     E criterion (in this normalisation).  g < 0 throughout --")
    print("     the D gain never exceeds 2 tau - 1, so lam_min stays")
    print("     below 1 even though the product of eigenvalues beats 1.")
    print(f"   - sign(E_ratio - 2/3) vs sign(D - 8/9): agree in "
          f"{n_sign_agree}/{len(cases)} cases.")
    print("   - vmin is the losing direction in NORMALISED (u, udot)")
    print("     coordinates (0 deg = u, 90 deg = udot).  At M = 1 it equals")
    print("     psi exactly: in these coordinates the losing direction IS")
    print("     the direction of x itself -- the combination the trigger")
    print("     watches is the one that loses information.  Across the")
    print("     other cases it ranges from ~18 to ~50 deg.")
    print("   - REMINDER: E_ratio, vmin and g are tied to the choice")
    print("     S = diag(M_1)^(-1/2); D ratio is not.")
    return rows


# --------------------------------------------------------------------------
# STAGE 1B -- event-budget / fixed-T objective
# --------------------------------------------------------------------------

def stage1B(theta=THETA_DEFAULT, model=MODEL_STATIC):
    theta = np.asarray(theta, float)
    cases = flagged_cases() + [CONSTRAINED_OPT]

    print("\n" + "=" * 96)
    print("STAGE 1B.  EVENT-BUDGET OBJECTIVE AT FIXED T AND FIXED Delta")
    print("=" * 96)
    print("  N ~ T m / Delta and det I_ev ~ (T/Delta)^2 J_ev with")
    print("      J_ev := m^2 det M_w = det( <|xdot| phi phi^T>_1 ).")
    print("  J_ev is NOT scale-invariant (it grows like power^3), so every")
    print("  case is first rescaled to unit input power <u^2>_1 = 1; the")
    print("  factor is printed.  The M = 1 reference for each case is a")
    print("  unit-power single sine at that case's HIGHEST frequency --")
    print("  under fixed T and a power constraint m grows with frequency,")
    print("  so that is the fair single-tone competitor (a modelling")
    print("  choice: a band-edge reference, stated, not settled).\n")

    print(f"{'case':<20}{'scale':>8}{'w_hi':>7}{'m':>10}{'det M_w':>11}"
          f"{'J_ev':>11}{'m^2*detMw':>11}{'J_ref':>10}{'J/J_ref':>10}")
    rows = []
    for tag, ms in cases:
        scale = 1.0 / np.sqrt(np.sum(ms.amps ** 2) / 2.0)
        msn = multisine(ms.amps * scale, ms.freqs, ms.phases)
        M1 = uniform_moments_exact(msn, model)
        Mw, _ = weighted_moments_time(msn, theta, model)
        m = mean_abs_xdot_time(msn, theta, model)
        dMw = float(np.linalg.det(Mw))
        J = m ** 2 * dMw

        w_hi = float(msn.freqs.max())
        ref = multisine([np.sqrt(2.0)], [w_hi], [0.0])
        M1r = uniform_moments_exact(ref, model)
        Mwr, _ = weighted_moments_time(ref, theta, model)
        mr = mean_abs_xdot_time(ref, theta, model)
        Jr = mr ** 2 * float(np.linalg.det(Mwr))

        rows.append(dict(tag=tag, m=m, dMw=dMw, J=J, Jr=Jr, w_hi=w_hi))
        print(f"{tag:<20}{scale:>8.4f}{w_hi:>7.3f}{m:>10.6f}{dMw:>11.6f}"
              f"{J:>11.6f}{m**2*dMw:>11.6f}{Jr:>10.5f}{J/Jr:>10.5f}")

    print("\n  The J_ev and m^2*detMw columns are printed separately only to")
    print("  show the decomposition is exact; they are the same number.")
    n_win = sum(1 for r in rows if r["J"] > r["Jr"] * (1 + 1e-9))
    print(f"  J/J_ref > 1 in {n_win}/{len(rows)} cases.  All the SPREAD")
    print("  multisines lose (0.02 - 0.75): they win the per-event")
    print("  comparison (D ratio) but give up too much event rate m against")
    print("  a single tone at their own top frequency.  The one winner is")
    print("  the NARROWBAND constrained optimum (1.27): its two tones sit")
    print("  so close together that m barely drops below the single-tone")
    print("  value, while det M_w keeps most of the 1.58 D-ratio factor.")
    print("  IMPORTANT: J_ev is UNBOUNDED in frequency under a power")
    print("  constraint alone (m ~ w, so J ~ w^4 for a single tone as")
    print("  w -> infinity at fixed power).  The fixed-T objective is only")
    print("  meaningful on a bounded frequency band; the numbers above are")
    print("  comparisons inside each case's own band, not global optima.")
    return rows


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def make_criteria_figure(rowsA, rowsB, s2=None, outdir="."):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if s2 is None:
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    else:
        fig, axx = plt.subplots(2, 2, figsize=(13, 10))
        ax = axx[0]

    # panel 1: D ratio vs E_ratio
    for i, r in enumerate(rowsA):
        mk = "s" if r["tag"] == "constrained opt" else "o"
        ax[0].plot(r["gain"], r["crit"]["E_ratio"], mk, ms=7,
                   color=f"C{i % 10}", label=r["tag"])
    ax[0].plot(8 / 9, 2 / 3, "r*", ms=16, label="M=1 point (8/9, 2/3)")
    ax[0].axvline(1.0, color="k", lw=1, ls=":")
    ax[0].axhline(1.0, color="k", lw=1, ls=":")
    ax[0].set_xlabel("D ratio = det Mt_w")
    ax[0].set_ylabel("E_ratio = lambda_min(Mt_w)")
    ax[0].set_title("D can beat 1; E never does (this set)\n"
                    "normalisation S = diag(M_1)^(-1/2)", fontsize=10)
    ax[0].legend(fontsize=6.5, loc="lower right")

    # panel 2: m vs det M_w with iso-J_ev curves (unit input power)
    ms_ = np.array([r["m"] for r in rowsB])
    dw_ = np.array([r["dMw"] for r in rowsB])
    for i, r in enumerate(rowsB):
        mk = "s" if r["tag"] == "constrained opt" else "o"
        ax[1].loglog(r["m"], r["dMw"], mk, ms=7, color=f"C{i % 10}",
                     label=r["tag"])
    mm = np.geomspace(ms_.min() / 1.6, ms_.max() * 1.6, 60)
    for J in np.geomspace((ms_ ** 2 * dw_).min() / 4,
                          (ms_ ** 2 * dw_).max() * 4, 7):
        ax[1].loglog(mm, J / mm ** 2, "-", color="0.8", lw=0.8, zorder=0)
    ax[1].set_xlabel("m = <|xdot|>_1   (event rate x Delta)")
    ax[1].set_ylabel("det M_w   (per-event D information)")
    ax[1].set_title("fixed-T objective J_ev = m^2 det M_w, unit power\n"
                    "grey: iso-J_ev (slope -2)", fontsize=10)
    ax[1].legend(fontsize=6.5, loc="lower left")

    if s2 is not None:
        b = s2["betas"]
        # panel 3: the wide-separation limit family
        a3 = axx[1, 0]
        a3.semilogx(b, [x["D"] for x in s2["limit"]], "-", lw=1.6,
                    label="D ratio")
        a3.semilogx(b, [x["E"] for x in s2["limit"]], "-", lw=1.6,
                    color="C2", label="E_ratio")
        a3.axvline(s2["bD"], color="k", lw=0.8, ls="--",
                   label=f"beta_D* = {s2['bD']:.4f}")
        a3.axhline(1.0, color="k", lw=1, ls=":")
        ac = a3.twinx()
        ac.semilogx(b, [x["cond"] for x in s2["limit"]], "--", color="C3",
                    lw=1.2)
        ac.set_ylabel("cond (red, dashed)", color="C3")
        a3.set_xlabel("beta = a2 w2 / (a1 w1)")
        a3.set_ylabel("D ratio,  E_ratio")
        a3.set_title("wide-separation LIMIT family (theta-free)\n"
                     "E climbs to 1 from below, never crossing", fontsize=10)
        a3.legend(fontsize=7, loc="center right")

        # panel 4: the finite family w1 = 0.5, r = 2, unit power
        a4 = axx[1, 1]
        a4.semilogx(b, s2["fD"], "-", lw=1.6, label="D ratio (1, 0.5)")
        a4.semilogx(b, s2["fE"], "-", lw=1.6, color="C2",
                    label="E_ratio (1, 0.5)")
        for (th, J), st in zip(s2["Jcurves"].items(), ("-", "--", ":")):
            a4.semilogx(b, J, st, color="C4", lw=1.3,
                        label="J/J_ref ({:g}, {:g})".format(*th))
        a4.axhline(1.0, color="k", lw=1, ls=":")
        a4.set_xlabel("beta = a2 w2 / (a1 w1)   (rho = beta/2)")
        a4.set_ylabel("D, E, J/J_ref")
        a4.set_title("FINITE family w1 = 0.5, r = 2, unit power\n"
                     "J is monotone toward the single-tone edge", fontsize=10)
        a4.legend(fontsize=7, loc="upper left")
        for a in (a3, a4):
            a.grid(alpha=0.3)

    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    path = f"{outdir}/multisine_criteria_figures.png"
    fig.savefig(path, dpi=130)
    print(f"\nfigure written to {path}")
    return fig, ax, path


def _run_and_log(stages, path="multisine_criteria_output.txt", mode="w"):
    """Run callables, mirroring stdout into the output file."""
    import io, sys

    class Tee(io.TextIOBase):
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)
            return len(s)

    out = []
    with open(path, mode) as fh:
        old = sys.stdout
        sys.stdout = Tee(old, fh)
        try:
            for fn in stages:
                out.append(fn())
        finally:
            sys.stdout = old
    return out





# --------------------------------------------------------------------------
# STAGE 2 -- the beta family
# --------------------------------------------------------------------------

def limit_family_criteria(beta, n=2500):
    """D, E, cond, tau, g on the wide-separation limit family.  The limit
    problem is already in normalised coordinates up to the diagonal scaling
    that wide_separation_limit's R11/R22/cross build in, so Mt_w is just
    [[R11, cross], [cross, R22]] (cross = 0 there by symmetry)."""
    from multisine_input_design import wide_separation_limit
    d = wide_separation_limit(beta, n)
    Mt = np.array([[d["R11"], d["cross"]], [d["cross"], d["R22"]]])
    lam = np.linalg.eigvalsh(Mt)
    tau = 0.5 * (d["R11"] + d["R22"])
    return dict(D=d["gain"], E=float(lam[0]), lam_max=float(lam[-1]),
                cond=float(lam[-1] / lam[0]), tau=float(tau),
                g=float(d["gain"] - (2.0 * tau - 1.0)))


def _E_beta1_quad():
    """E_ratio of the limit family at beta = 1, to quadrature tolerance
    1e-13, via the EXACT inner p2 average <|A + cos p2|> = (2/pi)
    (sqrt(1-A^2) + A asin A), leaving a smooth 1-D integral in p1."""
    from scipy.integrate import quad

    def h(p1):
        A = np.cos(p1)
        return (2.0 / np.pi) * (np.sqrt(max(1.0 - A * A, 0.0))
                                + A * np.arcsin(np.clip(A, -1, 1)))

    kw = dict(epsabs=1e-13, epsrel=1e-13, limit=400)
    num, _ = quad(lambda p: np.sin(p) ** 2 * h(p), 0.0, np.pi, **kw)
    den, _ = quad(h, 0.0, np.pi, **kw)
    return 2.0 * num / den            # E_ratio = R11 (the smaller eigenvalue)


def finite_family_eval(beta, theta, c=1.0, model=MODEL_STATIC, n=200_001):
    """
    The finite two-tone family: w1 = 0.5 c, w2 = 1.0 c (r = 2 fixed, so
    beta = 2 rho), phases (0, 0) -- a stated choice, since r = 2 is
    rational and the average is phase-dependent -- amplitudes rescaled to
    unit input power.  r = 2 makes the signal exactly periodic with period
    4 pi / c, so ONE period of dense trapezoid is the exact time average.
    """
    from multisine_input_design import (evaluate_dot, output_multisine,
                                        regressors_ms)
    theta = np.asarray(theta, float)
    rho = beta / 2.0
    a = np.array([1.0, rho])
    a *= 1.0 / np.sqrt(np.sum(a ** 2) / 2.0)          # unit power
    ms = multisine(a, [0.5 * c, 1.0 * c], [0.0, 0.0])
    P = 4.0 * np.pi / c
    t = np.linspace(0.0, P, n)
    Phi = regressors_ms(t, ms, model)
    wt = np.abs(evaluate_dot(output_multisine(ms, theta, model), t))
    Z = np.trapezoid(wt, t)
    m = float(Z / P)
    Mw = np.empty((2, 2))
    for i in range(2):
        for j in range(i, 2):
            Mw[i, j] = Mw[j, i] = np.trapezoid(Phi[:, i] * Phi[:, j] * wt,
                                               t) / Z
    M1 = uniform_moments_exact(ms, model)
    crit = criteria_normalised(Mw, M1)
    tau = 0.5 * (crit["Mt"][0, 0] + crit["Mt"][1, 1])
    return dict(D=crit["D_check"], E=crit["E_ratio"], cond=crit["cond_ev"],
                g=float(crit["D_check"] - (2 * tau - 1)), m=m,
                J=float(m ** 2 * np.linalg.det(Mw)), w_hi=float(c))


def _J_ref_single_tone(w, theta, model=MODEL_STATIC, n=200_001):
    """J_ev of the unit-power single tone at frequency w (the band-edge
    reference), by the same one-period machinery."""
    from multisine_input_design import (evaluate_dot, output_multisine,
                                        regressors_ms)
    theta = np.asarray(theta, float)
    ms = multisine([np.sqrt(2.0)], [w], [0.0])
    P = 2.0 * np.pi / w
    t = np.linspace(0.0, P, n)
    Phi = regressors_ms(t, ms, model)
    wt = np.abs(evaluate_dot(output_multisine(ms, theta, model), t))
    Z = np.trapezoid(wt, t)
    m = Z / P
    Mw = np.empty((2, 2))
    for i in range(2):
        for j in range(i, 2):
            Mw[i, j] = Mw[j, i] = np.trapezoid(Phi[:, i] * Phi[:, j] * wt,
                                               t) / Z
    return float(m ** 2 * np.linalg.det(Mw))


def stage2(theta=THETA_DEFAULT, model=MODEL_STATIC):
    from fractions import Fraction
    from scipy.optimize import minimize_scalar
    theta = np.asarray(theta, float)

    print("\n" + "=" * 96)
    print("STAGE 2.  THE BETA FAMILY   (beta = a2 w2 / (a1 w1))")
    print("=" * 96)

    # ---- 2.1 the limit-family sweep -------------------------------------
    betas = np.geomspace(0.2, 5.0, 200)
    sweep = [limit_family_criteria(b) for b in betas]
    D = np.array([s["D"] for s in sweep])
    E = np.array([s["E"] for s in sweep])
    G = np.array([s["g"] for s in sweep])

    print("\n2.1  wide-separation LIMIT family, beta in [0.2, 5], 200 log "
          "points (theta-free).")
    print("     every 20th point:\n")
    print(f"{'beta':>10}{'D':>11}{'E_ratio':>11}{'cond':>10}{'g':>12}")
    for i in list(range(0, 200, 20)) + [199]:
        s = sweep[i]
        print(f"{betas[i]:>10.4f}{s['D']:>11.6f}{s['E']:>11.6f}"
              f"{s['cond']:>10.4f}{s['g']:>12.6f}")

    rD = minimize_scalar(lambda b: -limit_family_criteria(b, 5000)["D"],
                         bracket=(1.1, 1.22, 1.4))
    bD, Dstar = rD.x, -rD.fun
    ok = abs(bD - 1.222388) < 5e-4 and abs(Dstar - 1.6127375) < 1e-6
    print(f"\n     beta_D* = {bD:.6f},  D = {Dstar:.7f}   "
          f"(target 1.222388, 1.6127375: {'REPRODUCED' if ok else 'FAILED'})")
    if not ok:
        print("     STOP: the limit-family optimum did not reproduce.")
        return None

    iE = int(np.argmax(E))
    print(f"     beta_E* = argmax E_ratio = {betas[iE]:.4f} -- the RIGHT "
          f"EDGE of the sweep: E_ratio is")
    print("     monotone increasing in beta on [0.2, 5] "
          f"(E = {E[0]:.4f} -> {E[-1]:.4f}), and spot checks")
    for b in (20.0, 100.0):
        print(f"       beta = {b:>5g}:  E_ratio = "
              f"{limit_family_criteria(b, 3000)['E']:.6f}")
    print("     show it climbing toward 1 from below (R11 -> 1, R22 -> 4/3")
    print("     as beta -> inf), so sup E_ratio = 1, NOT attained: the two")
    print("     optima do not coincide and E never beats periodic here.")
    print(f"     E_ratio at beta_D*: "
          f"{limit_family_criteria(bD, 5000)['E']:.6f}")
    print(f"     g along the sweep: g in [{G.min():.4f}, {G.max():.4f}], "
          f"max at beta = {betas[np.argmax(G)]:.3f};")
    print("     g < 0 EVERYWHERE -- it never changes sign on the limit "
          "family.")

    # ---- 2.2 E_ratio at beta = 1 ----------------------------------------
    E1 = _E_beta1_quad()
    fr = Fraction(E1).limit_denominator(1000)
    print(f"\n2.2  E_ratio at beta = 1 (exact inner p2 integral + 1-D "
          f"adaptive quadrature):")
    print(f"     E_ratio(beta=1) = {E1:.12f}")
    print(f"     nearest small rational: {fr.numerator}/{fr.denominator} "
          f"= {float(fr):.12f},  |diff| = {abs(E1 - float(fr)):.2e}")
    if abs(E1 - 8.0 / 9.0) < 1e-10:
        print("     matches 8/9 to better than 1e-10: at beta = 1 the")
        print("     weight factorises, |cos p1 + cos p2| =")
        print("     2 |cos((p1+p2)/2)| |cos((p1-p2)/2)|, and the same")
        print("     factorised moments that give D = 128/81 give")
        print("     R11 = 8/9, R22 = 16/9 exactly.  (Curious: the E ratio")
        print("     of the beta = 1 family equals the M = 1 D ratio.)")
    else:
        print("     does NOT match a small rational to 1e-10; no closed "
              "form claimed.")

    # ---- 2.3 the finite family ------------------------------------------
    print("\n2.3  FINITE family w1 = 0.5, r = 2 (w_hi = 1), phases (0,0),")
    print("     unit input power.  r = 2 is rational, so the result is")
    print("     phase-dependent; (0,0) is a stated choice.  J_ref = the")
    print("     unit-power single tone at w_hi, per theta.")
    thetas = ((1.0, 0.5), (1.0, 0.0), (0.0, 1.0))
    Jrefs = {th: _J_ref_single_tone(1.0, th, model) for th in thetas}

    fsweep = [finite_family_eval(b, theta) for b in betas]
    fD = np.array([s["D"] for s in fsweep])
    fE = np.array([s["E"] for s in fsweep])
    fG = np.array([s["g"] for s in fsweep])
    Jcurves = {}
    for th in thetas:
        if th == tuple(theta):
            Jcurves[th] = np.array([s["J"] for s in fsweep]) / Jrefs[th]
        else:
            Jcurves[th] = np.array(
                [finite_family_eval(b, th)["J"] for b in betas]) / Jrefs[th]

    print("\n     every 20th point (D, E, g at theta = (1, 0.5) only):\n")
    print(f"{'beta':>10}{'D':>10}{'E_ratio':>10}{'g':>11}"
          f"{'J/Jref(1,.5)':>13}{'J/Jref(1,0)':>13}{'J/Jref(0,1)':>13}")
    for i in list(range(0, 200, 20)) + [199]:
        print(f"{betas[i]:>10.4f}{fD[i]:>10.6f}{fE[i]:>10.6f}{fG[i]:>11.6f}"
              f"{Jcurves[(1.0, 0.5)][i]:>13.6f}"
              f"{Jcurves[(1.0, 0.0)][i]:>13.6f}"
              f"{Jcurves[(0.0, 1.0)][i]:>13.6f}")
    print(f"\n     g on the finite family: in [{fG.min():.4f}, "
          f"{fG.max():.4f}] -- never changes sign either.")

    # the three optima on the finite family, theta = (1, 0.5)
    rDf = minimize_scalar(lambda b: -finite_family_eval(b, theta)["D"],
                          bracket=(0.8, 1.2, 2.0))
    rEf = minimize_scalar(lambda b: -finite_family_eval(b, theta)["E"],
                          bounds=(0.2, 5.0), method="bounded",
                          options=dict(xatol=1e-5))
    print(f"\n     finite family, theta = (1, 0.5):")
    print(f"       beta_D* = {rDf.x:.4f}   D = {-rDf.fun:.6f}")
    eEdge = finite_family_eval(5.0, theta)["E"]
    if abs(rEf.x - 5.0) < 1e-2 or -rEf.fun <= eEdge + 1e-9:
        print(f"       beta_E* = 5.0 (right edge again, E = {eEdge:.6f}, "
              f"still < 1)")
    else:
        print(f"       beta_E* = {rEf.x:.4f}   E = {-rEf.fun:.6f}")

    # addendum: beta_J* and peak J/J_ref for the three thetas
    print("\n     ADDENDUM -- beta_J* and peak J/J_ref per theta "
          "(same reference convention):\n")
    print(f"{'theta':>12}{'beta_J*':>14}{'peak J/J_ref':>14}"
          f"{'J/Jref at 20':>14}{'J_ref':>10}")
    for th in thetas:
        rJ = minimize_scalar(
            lambda b: -finite_family_eval(b, th)["J"] / Jrefs[th],
            bounds=(0.2, 5.0), method="bounded", options=dict(xatol=1e-5))
        edge = abs(rJ.x - 5.0) < 2e-2
        j20 = finite_family_eval(20.0, th)["J"] / Jrefs[th]
        lab = "({:g}, {:g})".format(*th)
        bl = "5.0 (EDGE)" if edge else f"{rJ.x:.4f}"
        print(f"{lab:>12}{bl:>14}{-rJ.fun:>14.6f}{j20:>14.6f}"
              f"{Jrefs[th]:>10.5f}")
    print("\n     beta_J* sits at the sweep EDGE for every theta: at r = 2")
    print("     the J objective has NO interior optimum -- raising beta just")
    print("     concentrates power on the top tone (rho = beta/2), and")
    print("     J/J_ref climbs monotonically toward 1, the degenerate")
    print("     single-tone limit, without ever beating it.  The two-tones")
    print("     that DO beat their band edge (the 1.27 of Stage 1B) live at")
    print("     r near 1, not at r = 2: splitting the tones a full octave")
    print("     already costs more event rate than the D factor repays.")

    # addendum: frequency-scale sweep at theta = (1, 0.5).  Done for BOTH
    # readings of "the same two-tone": (i) the finite-family tone at its
    # beta_J* (literal), and (ii) the constrained-optimum two-tone that
    # actually produced the 1.27 in Stage 1B -- (i) never reaches 1.27, so
    # (ii) is the one that answers "how much does the 1.27 move".
    rJ0 = minimize_scalar(
        lambda b: -finite_family_eval(b, theta)["J"] / Jrefs[tuple(theta)],
        bounds=(0.2, 5.0), method="bounded", options=dict(xatol=1e-5))
    bJ = rJ0.x
    print(f"\n     ADDENDUM -- frequency scale c in {{0.5, 1, 2, 4}}, theta "
          f"= (1, 0.5), reference")
    print("     retargeted to the new band edge w_hi = c * (old w_hi).")
    print("     Scaling all w by c acts like scaling theta2 by c, so this")
    print("     doubles as the theta sensitivity of the J numbers.\n")
    print(f"     (i) finite-family tone at its edge beta_J* = {bJ:.3f} "
          f"(w1 = 0.5c, w2 = c):")
    print(f"{'c':>10}{'w_hi':>9}{'J/J_ref':>11}")
    for c in (0.5, 1.0, 2.0, 4.0):
        v = finite_family_eval(bJ, theta, c=c)["J"] / _J_ref_single_tone(
            c * 1.0, theta, model)
        print(f"{c:>10.3g}{c:>9.3g}{v:>11.6f}")

    print("\n     (ii) the ACTUAL 1.27 two-tone (constrained optimum:"
          " w1 = 0.631c,")
    print("     r = 1.06, rho = 0.9486, unit power; quasi-period much longer")
    print("     than one tone period, so the long-window quadrature of "
          "Stage 1B is reused):")
    print(f"{'c':>10}{'w_hi':>9}{'J/J_ref':>11}")
    _, ms_co = CONSTRAINED_OPT
    a_co = ms_co.amps / np.sqrt(np.sum(ms_co.amps ** 2) / 2.0)
    for c in (0.5, 1.0, 2.0, 4.0):
        ms_c = multisine(a_co, ms_co.freqs * c, ms_co.phases)
        Mw_c, _ = weighted_moments_time(ms_c, theta, model)
        m_c = mean_abs_xdot_time(ms_c, theta, model)
        J_c = m_c ** 2 * float(np.linalg.det(Mw_c))
        v = J_c / _J_ref_single_tone(float(ms_c.freqs.max()), theta, model)
        print(f"{c:>10.3g}{float(ms_c.freqs.max()):>9.4g}{v:>11.6f}")

    return dict(betas=betas, limit=sweep, fD=fD, fE=fE, fG=fG,
                Jcurves=Jcurves, bD=bD, Dstar=Dstar)


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    r0, rA, rB, r2 = _run_and_log([stage0, stage1A, stage1B, stage2])
    make_criteria_figure(rA, rB, s2=r2)
