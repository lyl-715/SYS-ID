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


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    stage0()
