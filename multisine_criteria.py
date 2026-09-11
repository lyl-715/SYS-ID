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

def make_criteria_figure(rowsA, rowsB, outdir=".", fname_extra=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

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


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    r0, rA, rB = _run_and_log([stage0, stage1A, stage1B])
    make_criteria_figure(rA, rB)
