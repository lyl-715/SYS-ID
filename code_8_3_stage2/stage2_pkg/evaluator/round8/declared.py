"""Round 8: the declared knowledge (spec reports_CN/plan_round8_CN.md v3.1, frozen; §1 "事先声明的知识").

Everything here may be read by the estimation and design functions. Nothing here holds, or is computed from, the true
parameter vector theta_0 or the nominal output: model structure x_k = h_k^T theta with the single 5-s regressor matrix H
(frequencies w, uniform input u_unif, clock), sigma and Gaussianity, the constant threshold Delta*, the budget constant B,
the action set, grid step and support rule, the design rule lambda = 4 gbar, the backup vector theta_bk = [1, 0, ..., 0], the
estimator's parameter scaling s = [2, 1, ..., 1].

The values are literals copied from the frozen sources (mc83.py setup block, mc85_scan.py, density_recursion_v1.py,
evaluator_window.py, trigger_dp.py, round 6/7 records); checks_r8.py asserts each one bit for bit against those sources
(which hold theta_0 and are therefore never imported by this module or by the design module).

H itself is built by the verbatim mc85_scan.regressors (extracted into r8k_dr.py by build_kernels.py), on first use.
"""
import numpy as np

TWO_PI = 2*np.pi

# --- clock, frequencies, input (mc83.py setup lines 3-5) -------------------------------------------------------------
T, dt = 5.0, 1e-3; K = int(T/dt)
t = np.arange(1, K+1)*dt
w = np.array([3*np.sqrt(2), 3*np.pi, 12.0, 12.3]); a0 = 1.0
A_U = np.full(4, 0.5); CHI_U = np.zeros(4)               # u_unif (evaluator config.json input U; chi reduced mod 2 pi on use)

# --- noise and trigger (mc85_scan.py / density_recursion_v1.py) --------------------------------------------------------
D = 0.002384694482635785                                  # Delta*
s = D/20                                                  # sigma (Gaussian, iid)
B = 316.8921168493721                                     # frozen budget constant (round2a/config_recheck.json budget_B)

# --- action set, grid, support rule (evaluator_window.py, trigger_dp.py) ----------------------------------------------
N_GRID = 24                                               # design grid h = sigma/24
SOD_M = 20*N_GRID                                         # (480, 480) grid units = constant Delta*
STEP_M = 2*N_GRID                                         # action spacing 48 grid units = 2 sigma
N_ACT1 = 20                                               # M^+- in {48, 96, ..., 960}
X24 = N_ACT1*STEP_M                                       # largest half-width, 960 grid units
R_SUP = 88*s                                              # general-path support half-width
MARGIN_MULT = 8                                           # grid range [min x - 8 Delta, max x + 8 Delta]

# --- design rule (spec §1.3): lambda = 4 gbar, gbar = mean_{k>=2} gamma_k ----------------------------------------------
def design_lambda(gbar):
    """lambda = 4 gbar, formed in floating point as 0.8 * (5 gbar): the exact arithmetic of the round-6 scan run that defined
    T4 (round7/tables.py; 4.0 * gbar differs from it in the last bit at the nominal gbar). DECISION A-3."""
    return 0.8*(5.0*gbar)

# --- estimator constants (spec §1.2) -----------------------------------------------------------------------------------
SCALE = np.r_[2.0, np.ones(8)]                            # vartheta = theta / s
THETA_BK = np.r_[1.0, np.zeros(8)]                        # backup parameter vector
N_PAR = 9


def x_polar(theta):
    """nominal output x_k = h_k^T theta for the uniform input, evaluated in the polar form of the declared model structure:
    h_k^T theta = theta_1 a0 + sum_m a_m |g_m| cos(w_m t_k + chi_m + arg g_m), g_m = theta_{2m} + i theta_{2m+1} -- the same
    arithmetic (and summation order) as mc85_scan.output / mc83.py, so that a vector equal to the nominal frequency response
    gives the round-4 nominal output bit for bit (DECISION A-2)."""
    th = np.asarray(theta, float)
    g = th[1::2]+1j*th[2::2]
    a = A_U; chi = np.mod(CHI_U, TWO_PI)
    return th[0]*a0 + sum(a[m]*abs(g[m])*np.cos(w[m]*t+chi[m]+np.angle(g[m])) for m in range(4))


_H = None
def H():
    """regressor matrix (K, 9), h_k = [a0, a_m cos psi_m, -a_m sin psi_m], by the verbatim mc85_scan.regressors."""
    global _H
    if _H is None:
        import r8k_dr
        _H = r8k_dr.regressors(A_U, np.mod(CHI_U, TWO_PI), t)
        _H.setflags(write=False)
    return _H
