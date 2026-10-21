"""Round 2b, shared by mdp1d_newgrid.py and mdp2d_estimate.py: the round-2a discrete model (round2a/mdp_estimate.py) on the new
grid of spec reports_CN/plan_round2b_CN.md v3.2 §2, and the golden-section outer loop with an on-disk cache.

Reuse, not rewrite: round2a/mdp_estimate.py is imported unchanged; its module constants are set to the new grid before any call
(its functions read N_SIG, G, M, L, ORDER from the module at call time), so model(), _q_action(), _max_step(), _init_value(),
_forward_step(), initial_mu(), forward(), backward_fixed() and dual_value() are the round-2a code on
    delta/sigma = -60 + i/30, i = 0..3600 (G = 3601; [-Delta - 40 sigma, Delta + 40 sigma], h = sigma/30), M = 600 (Delta = 600 h),
    velocities v = l h, |l| <= L = 137 (137 h = 4.5667 sigma), 275 values; ORDER = smallest |v| first, then +v before -v.
c = mc83.cfun (via density_recursion_v1, as in round 2a), q = Phi(alpha) + Phi(-beta), P = 1 - q, send kernel = discretised Gaussian
masked by |j - i| >= M and renormalised per row (left prefix + right suffix sums)."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
sys.path.insert(0, EV); sys.path.insert(0, os.path.join(EV, 'round2a'))
import numpy as np
import mdp_estimate as mdp
import density_recursion_v1 as dr

N_SIG, HALF, L = 30, 60, 137
mdp.N_SIG = N_SIG; mdp.HALF = HALF; mdp.G = 2*HALF*N_SIG+1; mdp.M = 20*N_SIG; mdp.L = L
mdp.ORDER = np.array([0]+[s*l for l in range(1, L+1) for s in (1, -1)], np.int64)
G, M, ORDER, K = mdp.G, mdp.M, mdp.ORDER, dr.K
NL = 2*L+1; LH = L                                      # column index c = l + L
LAM_BUDGET = mdp.LAM_BUDGET                             # 315.892 = B - 1 (spec §1.3, as round 2a)
LAM_STAR_2A = 1.8335414994039494                        # round2a/logs/mdp_estimate.log: lambda* (full precision, archived)
S_DUAL_2A = 1210.9563921988488                          # round2a/logs/mdp_estimate.log: S~_dual on the old grid (h = sigma/24, L = 110)
B_FULL = 316.8921168493721                              # round2a/config_recheck.json budget_B (E[N] of the uniform input)
I0 = HALF*N_SIG                                         # index of delta = 0
BAND = (18*N_SIG, 19*N_SIG)                             # T3: 18 <= |delta|/sigma <= 19  <->  540 <= |i - I0| <= 570 (integer test)
LOW = N_SIG//2                                          # T3: |v| <= 0.5 sigma  <->  |l| <= 15 (integer test)
assert G == 3601 and M == 600 and NL == 275 and LOW == 15 and BAND == (540, 570)
LOGS = os.path.join(HERE, 'logs'); RES = os.path.join(HERE, 'results')
os.makedirs(LOGS, exist_ok=True); os.makedirs(RES, exist_ok=True)


def band_mask():
    d = np.abs(np.arange(G)-I0); return (d >= BAND[0]) & (d <= BAND[1])


def golden(F, eps, log, lo=0.0, hi=20.0):
    """round-2a golden section (round2a/mdp_estimate.py __main__, same arithmetic) on [lo, hi] until hi - lo <= eps.
    Returns the evaluations of this run, the minimiser over them and the final bracket."""
    evals = {}
    def Fw(lam):
        v = F(float(lam)); evals[float(lam)] = v; return v[0]+LAM_BUDGET*float(lam)
    gr = float((np.sqrt(5.0)-1)/2)
    cc = hi-gr*(hi-lo); dd = lo+gr*(hi-lo); Fc = Fw(cc); Fd = Fw(dd)
    while hi-lo > eps:
        if Fc < Fd: hi, dd, Fd = dd, cc, Fc; cc = hi-gr*(hi-lo); Fc = Fw(cc)
        else: lo, cc, Fc = cc, dd, Fd; dd = lo+gr*(hi-lo); Fd = Fw(dd)
    lam_s = min(evals, key=lambda k: evals[k][0]+LAM_BUDGET*k)
    return evals, lam_s, evals[lam_s][0]+LAM_BUDGET*lam_s, lo, hi


class Cache:
    """per-lambda results on disk (one JSON line per evaluation, key = repr(lambda)); a killed outer loop resumes from here and the
    deterministic golden section replays the same lambdas."""
    def __init__(self, path):
        self.path = path; self.d = {}
        if os.path.exists(path):
            good = []
            for ln in open(path):
                try: r = json.loads(ln); self.d[r['lam']] = r; good.append(ln if ln.endswith('\n') else ln+'\n')
                except json.JSONDecodeError: pass
            open(path, 'w').write(''.join(good))
    def get(self, lam): return self.d.get(repr(float(lam)))
    def put(self, lam, rec):
        rec = dict(rec, lam=repr(float(lam))); self.d[rec['lam']] = rec
        with open(self.path, 'a') as f: f.write(json.dumps(rec)+'\n'); f.flush(); os.fsync(f.fileno())


def uniform_open_loop():
    """spec §3.3: the uniform input's true increments x_(k+1) - x_k rounded to multiples of h. Returns l_k for k = 1..K-1 (array
    index k-1), the max rounding error in units of h and the true increments in units of h."""
    xU, _ = dr.input_arrays('U'); hsig = dr.s/N_SIG; inc = np.diff(xU)/hsig
    l = np.rint(inc).astype(np.int64)
    return l, float(np.abs(inc-l).max()), inc


def loadavg(): return '/'.join(f'{v:.2f}' for v in os.getloadavg())
