"""Round 19: the complete two-stage experiment with the one-parameter rule (prompt
reports_CN/prompt_round19_two_stage_rule_EN.md). Input U, B = 316.8921168493721. Branch round19-two-stage-rule, started from
round18-accumulated-information @ 345b520.

Stage 1 (steps 1..K1): constant SoD, window r -/+ Delta*            (round18/acc18.window 'SOD').
Stage 2 (steps K1+1..K): the one-parameter rule with count feedback (round18/acc18.window 'RULE_FB' = round15/mismatch15.run_job
   'fb' branch), with the true output x_k replaced by the prediction xbar_k = h_k^T theta_tilde and the plan generalised from
   1 + q_t (k - 1) to N_1 + q_2 (k - K1):
      u_k = e_0 + (N_k - (N_1 + q_2 (k - K1)))/(2 q_2 T),  e_k = clip(u_k, 0.5, 4.0),  q_2 = (B - 3 - N_1)/(K - K1),
      lo = min(xbar_k - e_k s, r - s),  hi = max(xbar_k + e_k s, r + s);  e_0 = 1.577, T = 100.
   With K1 = 1, N_1 = 1 this is the existing form exactly (same floating-point expression).
Information: c_k = sum(dr.cfun((lo - x_k)/s, (hi - x_k)/s)) with the TRUE x_k (acc18.batch_A), c_1 = 1; per run
   J_cond = s^-2 sum_k c_k h_k h_k^T; J(config) = s^-2 sum_k wbar_k h_k h_k^T, wbar_k = mean over replications of c_k.
Estimators (round 8b, unchanged): stage 1 estimator_r8b.estimate_stage1 (LS start, one restart, Newton-decrement acceptance,
   fallback theta_LS); final estimator_r8b.estimate_joint on the whole record (start theta_tilde, one restart, same acceptance,
   fallback theta_tilde). Likelihood data carry the transmitted pairs and the windows of the silent steps as the receiver
   reconstructs them from the record (receiver_windows; checked bit for bit against the sensor's windows in every run).

Array layout as acc18.batch_A: per-step arrays are (K, R) (step index p = k - 1, replication in the column).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
R18 = os.path.join(EV, 'round18')
if R18 not in sys.path: sys.path.insert(0, R18)
import numpy as np
import acc18 as a                              # imports round15/mismatch15 -> round13/eval13 (input U), round8 estimator, round8b

m15, e8, e8b, dk, dr = a.m15, a.e8, a.e8b, a.dk, a.dr
s, K, B, P = a.s, a.K, a.B, a.P
XK, HK, HH = a.XK, a.HK, a.HH
D_STAR = a.D_STAR
THETA_STAR = a.THETA_STAR
I_SOD = 197.20097668224076
I_FULL = 219.00850592919505
assert a.I_SOD == I_SOD and B == 316.8921168493721 and K == 5000 and P == 9 and D_STAR == 0.002384694482635785
assert s == D_STAR/20 and np.array_equal(dk.H(), HK)
E0 = 1.577
FB_T, FB_M = 100, 3
E_LO, E_HI = 0.5, 4.0
assert (a.FB_T, a.FB_M, a.E_LO, a.E_HI) == (FB_T, FB_M, E_LO, E_HI)

NOISE_SEED0 = 190000000          # replication i: default_rng(190000000 + i).standard_normal(K)
S1_SEED0 = 191000000             # stage-1 CML restart seed 191000000 + i
FJ_SEED0 = 192000000             # final (joint) CML restart seed 192000000 + i
CHK_SEED0 = 193000000            # checks (a), (b): noise of trajectory i
BOOT_SEED = 194000000
N_BOOT = 2000


def noise(i, seed0=NOISE_SEED0):
    """the K standard normals of replication i: value 0 for z_1, value p for step p + 1 (as acc18.noise_B)."""
    return np.random.default_rng(seed0+i).standard_normal(K)

def noise_matrix(idx, seed0=NOISE_SEED0):
    """(K, R) noise of the replications idx (column j = replication idx[j])."""
    return np.column_stack([noise(int(i), seed0) for i in idx])

def predict(theta):
    """the receiver's (and the sensor's) prediction xbar = H theta (K,), one gemv for one parameter vector."""
    return HK@np.asarray(theta, float)

def logdet(M):
    return a.logdet(M)

def J_of_weights(W):
    return a.J_of_weights(W)


# --- the sensor: vectorised over replications -----------------------------------------------------------------------------
class Sim:
    """the sensor-side simulation of R replications (columns). Steps are advanced in order; sod() and rule() continue from the
    current step with the current reference r and count N."""
    def __init__(self, NZ):
        R = NZ.shape[1]; self.R = R
        self.z = XK[:, None]+s*NZ
        self.ev = np.zeros((K, R), bool); self.ev[0] = True
        self.lo = np.full((K, R), np.nan); self.hi = np.full((K, R), np.nan)
        self.C = np.empty((K, R)); self.C[0] = 1.0
        self.e = np.full((K, R), np.nan); self.u = np.full((K, R), np.nan)
        self.r = self.z[0].copy(); self.N = np.ones(R); self.S = np.ones(R)
        self.Wk = np.zeros(K); self.Wk[0] = float(R)
        self.p = 1
        self.K1 = None; self.N1 = None; self.q2 = None

    def copy(self):
        o = Sim.__new__(Sim)
        for k, v in self.__dict__.items():
            setattr(o, k, v.copy() if isinstance(v, np.ndarray) else v)
        return o

    def _step(self, p, lo, hi):
        assert p == self.p
        cev, cno = dr.cfun((lo-XK[p])/s, (hi-XK[p])/s); c = cev+cno
        self.C[p] = c; self.Wk[p] = c.sum(); self.S += c
        zk = self.z[p]
        snd = (zk <= lo) | (zk >= hi)
        self.ev[p] = snd; self.lo[p] = lo; self.hi[p] = hi
        self.N += snd; self.r = np.where(snd, zk, self.r)
        self.p += 1

    def sod(self, p_end):
        """constant SoD at step indices self.p .. p_end - 1 (acc18.window 'SOD')."""
        for p in range(self.p, p_end):
            r = self.r
            self._step(p, r-D_STAR, r+D_STAR)

    def rule(self, p_end, xbar, e0, fb):
        """the one-parameter rule at step indices self.p .. p_end - 1. xbar: (K,) (same prediction for every replication) or
        (K, R). fb True: count feedback with plan N_1 + q_2 (k - K1), N_1 = count before this call, K1 = steps before it."""
        K1 = self.p; N1 = self.N.copy(); q2 = (B-FB_M-N1)/(K-K1)
        self.K1, self.N1, self.q2 = K1, N1, q2
        for p in range(self.p, p_end):
            k = p+1
            if fb:
                u = e0+(self.N-(N1+q2*(k-K1)))/(2.0*q2*FB_T)
                e = np.clip(u, E_LO, E_HI); self.u[p] = u
            else:
                e = e0
            self.e[p] = e
            x = xbar[p]
            lo = np.minimum(x-e*s, self.r-s); hi = np.maximum(x+e*s, self.r+s)
            self._step(p, lo, hi)

    def J_cond(self):
        """per-replication J_cond flattened (R, 81) (acc18.batch_A: (C.T @ HH)/s^2)."""
        return (self.C.T@HH)/s**2


# --- the receiver: windows reconstructed from the record ----------------------------------------------------------------------
def receiver_windows(ev, zt, K1, xbar, e0, fb, stage2):
    """(lo, hi) at every step index p >= 1 (nan at p = 0) from the record (ev, zt) of ONE replication: r_k = the last
    transmitted value before k (estimator_r8.references); steps p < K1: SoD r -/+ Delta*; steps p >= K1: the rule (stage2 True)
    with xbar and e_k from the counts in the record, or SoD (stage2 False, constant SoD throughout)."""
    ev = np.asarray(ev, bool)
    r = e8.references(ev, zt)
    lo = np.full(K, np.nan); hi = np.full(K, np.nan)
    p1 = np.arange(1, K1) if stage2 else np.arange(1, K)
    lo[p1] = r[p1]-D_STAR; hi[p1] = r[p1]+D_STAR
    if stage2:
        Nb = np.concatenate(([0.0], np.cumsum(ev)[:-1].astype(float)))      # transmissions before step index p
        p2 = np.arange(K1, K); k = p2+1
        N1 = Nb[K1]; q2 = (B-FB_M-N1)/(K-K1)
        if fb:
            u = e0+(Nb[p2]-(N1+q2*(k-K1)))/(2.0*q2*FB_T)
            e = np.clip(u, E_LO, E_HI)
        else:
            e = e0
        x = xbar[p2]; rr = r[p2]
        lo[p2] = np.minimum(x-e*s, rr-s); hi[p2] = np.maximum(x+e*s, rr+s)
    return lo, hi

def lik_data(ev, zt, lo_n, up_n):
    """likelihood data with the fields estimator_r8.Objective / ls_init / stack read (He, ze, Hn, lo, up, N); as acc18.stage_data,
    for a record of length len(ev) (stage 1: K1 steps, whole record: K steps)."""
    n = len(ev); ev = np.asarray(ev, bool)
    d = e8.StageData.__new__(e8.StageData)
    Hm = dk.H()[:n]
    d.ev = ev; d.N = int(ev.sum()); d.He = Hm[ev]; d.ze = zt[ev]; d.Hn = Hm[~ev]
    d.lo = np.asarray(lo_n, float); d.up = np.asarray(up_n, float)
    assert d.Hn.shape[0] == d.lo.size == d.up.size
    assert np.all(np.isfinite(d.ze)) and np.all(np.isfinite(d.lo)) and np.all(np.isfinite(d.up))
    return d


# --- estimator jobs (one replication each; run in a Pool) ----------------------------------------------------------------------
def _att(att):
    return dict(accept=bool(att['accept']['supp']), success=bool(att['success']), nit=int(att['nit']), decr=float(att['decrement']),
                msg=str(att['message']))

def stage1_job(args):
    """stage-1 CML of one replication from its stage-1 record (steps 1..K1)."""
    i, K1, ev1, z1, lo_s, hi_s = args
    t0 = time.perf_counter()
    ev1 = np.asarray(ev1, bool); zt = np.where(ev1, z1, np.nan)
    r = e8.references(ev1, zt); kk = np.flatnonzero(~ev1)
    lo = r[kk]-D_STAR; hi = r[kk]+D_STAR
    okw = bool(np.array_equal(lo, lo_s[kk]) and np.array_equal(hi, hi_s[kk]))
    d = lik_data(ev1, zt, lo, hi)
    res = e8b.estimate_stage1(d, S1_SEED0+i)
    a1, a2 = res['attempts']
    return dict(i=int(i), K1=int(K1), N1=int(ev1.sum()), windows_ok=okw, theta=np.asarray(res['theta'], float),
                theta_LS=np.asarray(res['theta_LS'], float), LS_rank=int(res['LS_rank']), LS_usable=bool(res['LS_usable']),
                converged=bool(res['converged']), used=res['used'], fallback=res['fallback'], att1=_att(a1), att2=_att(a2),
                wall_s=time.perf_counter()-t0)

def final_job(args):
    """final CML (estimator_r8b.estimate_joint) of one replication on its whole record, windows reconstructed by the receiver.
    mode: 'est' (stage 2 with xbar = predict(theta_tilde)), 'true' (stage 2 with xbar = XK), 'sod' (constant SoD throughout)."""
    tag, i, K1, mode, ev, z, theta_tilde, lo_s, hi_s = args
    t0 = time.perf_counter()
    ev = np.asarray(ev, bool); zt = np.where(ev, z, np.nan)
    if mode == 'sod':
        lo, hi = receiver_windows(ev, zt, K, None, None, False, False)
    else:
        xbar = XK if mode == 'true' else predict(theta_tilde)
        lo, hi = receiver_windows(ev, zt, K1, xbar, E0, True, True)
    kk = np.flatnonzero(~ev)
    okw = bool(np.array_equal(lo[kk], lo_s[kk]) and np.array_equal(hi[kk], hi_s[kk]))
    d = lik_data(ev, zt, lo[kk], hi[kk])
    res = e8b.estimate_joint([d], np.asarray(theta_tilde, float), FJ_SEED0+i)
    a1, a2 = res['attempts']
    return dict(tag=tag, i=int(i), windows_ok=okw, theta=np.asarray(res['theta_final'], float), converged=bool(res['converged']),
                used=res['used'], fallback=res['fallback'], att1=_att(a1), att2=_att(a2), wall_s=time.perf_counter()-t0)


# --- statistics -------------------------------------------------------------------------------------------------------------
def pred_errors(theta_tilde, K1):
    """RMS and max |prediction error| over all K steps in units of sigma (mismatch15.pred_error), and the RMS over stage 2."""
    delta = np.asarray(theta_tilde, float)-THETA_STAR
    rms, mx = m15.pred_error(delta)
    d2 = HK[K1:]@delta/s
    return rms, mx, float(np.sqrt(np.mean(d2*d2)))

def acc_stats(th, J):
    """acc18.acc_stats (bias in CR sd, RMSE, Mahalanobis norm, D-hat, -log det cov) plus I - D-hat = log det(M-hat J)."""
    st = a.acc_stats(np.asarray(th, float), J)
    st['I'] = float(logdet(J)); st['I_minus_Dhat'] = st['I']-st['D_hat']
    st['var_ratio_gm'] = float(np.exp(st['I_minus_Dhat']/P)); st['sd_ratio_gm'] = float(np.exp(st['I_minus_Dhat']/(2*P)))
    return st

def D_hat_of(errs):
    E = np.asarray(errs, float)
    return float(-logdet(E.T@E/E.shape[0]))

def boot_counts(R, nboot=N_BOOT, seed=BOOT_SEED):
    """(nboot, R) multinomial resampling counts (indices drawn with replacement, then counted)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, R, size=(nboot, R))
    cnt = np.zeros((nboot, R))
    for b in range(nboot): cnt[b] = np.bincount(idx[b], minlength=R)
    return cnt

def logdet_rows(Jflat):
    """log det of each row of an (n, 81) array of flattened 9 x 9 matrices."""
    M = Jflat.reshape(-1, P, P)
    sg, ld = np.linalg.slogdet(M)
    assert np.all(sg > 0)
    return ld
