"""Round 2c: direct log-det dual prototype with two fixed A matrices (spec reports_CN/plan_round2c_CN.md, frozen, commit a716952).

Everything this file produces is an estimate on the discrete model (估算); nothing is certified.

Theory (spec §1). h_k(u) = T(u) h_{U,k}, T(u) = blockdiag(1, 2 a_1 R(chi_1), ..., 2 a_4 R(chi_4)), R(chi) = [[cos chi, sin chi],
[-sin chi, cos chi]], h_{U,k} = mc83.py's H (a_m = 1/2, chi_m = 0). M(w) = sigma^-2 sum_k w_k h_{U,k} h_{U,k}^T, J(u) = T M(w(u)) T^T.
For A > 0: log det M <= tr(A M) - log det A - 9, with equality iff A = M^-1. gamma_k = sigma^-2 h_{U,k}^T A h_{U,k}; tr(A M(w)) =
sum_k w_k gamma_k, w_1 = 1. The DP of round 2b with the per-step reward gamma_k c(delta) - lambda q(delta) gives V~_{A,lambda} and
    D(A, lambda) = -log det A - 9 + gamma_1 + lambda (B - 1) + V~_{A,lambda},   B - 1 = 315.892.
Two models on the same grid (h = sigma/30, L = 137): 2-D acceleration-constrained (round2b/mdp2d_estimate.py, Delta j in {-1, 0, 1})
and 1-D unconstrained (round2a/mdp_estimate.py with the grid of round2b/grid30.py).

Reuse, not rewrite: the kernels of round2b/mdp2d_estimate.py (_q_all, _wmax, _fwd, _apply_codes, init_value, open_loop_codes); the
round-2a functions via round2b/grid30.py (model, _max_step, _q_action, _init_value, initial_mu, _forward_step); golden() and Cache
of grid30.py; evaluator_fast.evaluate_fast (B, n = 24) for the weights; round2a/b5_conversion.py (bound_namespace, solve: route-1
exchange + certificate_full, cap 1) for U(S). The DP drivers below are the round-2b / round-2a drivers with one change: the reward
of step k is gamma_k c - lambda q instead of c - lambda q.

Indexing: array index p = 0..K-1 <-> step k = p + 1 (t_k = k dt), as in the evaluator (W[0] = 1 is the forced first sample);
DP step k = 2..K uses gamma[k-1].

Stages (python3 logdet_dual.py <stage>, from this directory; each appends to its own log in logs/):
  construct                                    Part A: A1 regressors and T(u), A2 weights, A3 M, A_i, gamma -> logs/part_a.log
  outer --A {1,2} --model {2d,1d} [--threads]  Part C1 for one case (refuses to run unless results/checks.json says all passed)
                                               -> logs/part_c1.log, logs/gold_<model>_A<i>.jsonl (per-lambda archive), results/
  sroute                                       Part C2: U(619.1292140245644; u_unif), route-1 solver, cap 1 -> logs/part_c2.log
  table                                        Part C3 from results/ -> logs/part_c3.log, results/part_c3.json"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EV, 'round2b'))
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
import grid30 as g
from grid30 import mdp, dr, G, M, K, NL, LH
import mdp2d_estimate as m2
import evaluator_fast as ef
from numba import set_num_threads, get_num_threads

SIG = dr.s; NPAR = 9
BM1 = g.LAM_BUDGET                          # 315.892 = B - 1 (spec §1.4; rounds 2a/2b, round2b DECISION A-1)
LD_JC = 197.197                             # log det J_C (spec §1.4, §1.5; as round2a/b5_conversion.LD_REF)
EPS = 4e-6                                  # epsilon_lambda (spec §2; frozen, no refinement rule this round)
SLOPE = 4684.0                              # |dD/dlambda| <= 4684 (spec §2, "同前")
HI_MULT = 20.0                              # lambda in [0, 20 gamma_bar] (spec §2)
LAM_2B, V_2B, S_2B = 0.9381038031641779, 321.78972743542596, 619.1292140245644   # round2b/logs/gold_2d.jsonl:21; results/mdp2d_outer_eps0.0025.json
LAM_1D_2B, V_1D_2B = 1.9185615985194624, 603.7566183574543                         # round2b/logs/gold_1d.jsonl:17 (1-D lambda*, extra regression)
CFG2A = json.load(open(os.path.join(EV, 'round2a', 'config_recheck.json')))
R7 = json.load(open(os.path.join(EV, 'round2a', 'results', 'recheck7.json')))
IDS = ['UNIF']+[c['id'] for c in CFG2A['candidates']]  # U and the seven round-2a candidates
UHAT = 'NM-none-P5514'
SRC = {1: UHAT, 2: 'UNIF'}                  # A_1 = M(w(u_hat))^-1, A_2 = M(w_U)^-1
LOGS = os.path.join(HERE, 'logs'); RES = os.path.join(HERE, 'results')
os.makedirs(LOGS, exist_ok=True); os.makedirs(RES, exist_ok=True)


# --- Part A: construction ----------------------------------------------------------------------------------------------------------
def evname(i): return 'U' if i == 'UNIF' else i

def register_candidates():
    """the seven candidates into the evaluator's in-memory input table only (as round2a/recheck7.py); no file is modified."""
    for c in CFG2A['candidates']: dr.CFG['inputs'][c['id']] = dict(a=c['a'], chi=c['chi'])

def ac_of(i):
    e = dr.CFG['inputs'][evname(i)]
    return np.asarray(e['a'], float), np.mod(np.asarray(e['chi'], float), dr.TWO_PI)

def h_uniform():
    """mc83.py's H (a_m = 1/2, chi_m = 0, a0 = 1), exec'd into density_recursion_v1 by its setup block; K x 9."""
    return np.ascontiguousarray(dr.H)

def T_of(a, chi):
    T = np.zeros((NPAR, NPAR)); T[0, 0] = 1.0
    for m in range(4):
        c_, s_ = np.cos(chi[m]), np.sin(chi[m])
        T[1+2*m:3+2*m, 1+2*m:3+2*m] = 2*a[m]*np.array([[c_, s_], [-s_, c_]])
    return T

def M_of(HU, w):
    """M(w) = sigma^-2 sum_k w_k h_U,k h_U,k^T, the evaluator's J assembly (density_recursion_v1.evaluate) with H_U."""
    return (HU*w[:, None]).T@HU/SIG**2

def factor(Mx):
    """lower Cholesky factor of M and log det M = 2 sum log diag."""
    Lc = cholesky(Mx, lower=True)
    return Lc, 2.0*float(np.log(np.diag(Lc)).sum())

def gamma_of(Lc, HU):
    """gamma_k = sigma^-2 h_U,k^T M^-1 h_U,k = sigma^-2 |L^-1 h_U,k|^2 (triangular solve; M^-1 is not formed)."""
    V = solve_triangular(Lc, HU.T, lower=True)
    return np.einsum('ij,ij->j', V, V)/SIG**2

def load_A(i):
    """(M_i, its Cholesky factor, log det M_i = -log det A_i, A_i, gamma^(i)) from results/."""
    Mi = np.load(os.path.join(RES, f'M_A{i}.npy')); Lc, ld = factor(Mi)
    return Mi, Lc, ld, np.load(os.path.join(RES, f'A{i}.npy')), np.load(os.path.join(RES, f'gamma_A{i}.npy'))

def stage_construct(log):
    register_candidates(); ef._warmup()
    HU = h_uniform(); a_u, chi_u = np.full(4, 0.5), np.zeros(4)
    log(f"A1 h_U = mc83.py H via density_recursion_v1 (exec of mc85_scan.py's setup block): shape {HU.shape}; equals regressors(1/2, 0, t) "
        f"bitwise: {np.array_equal(HU, dr.regressors(a_u, chi_u, dr.t))}; equals the evaluator's input_arrays('U') H bitwise: "
        f"{np.array_equal(HU, dr.input_arrays('U')[1])}; a0 = {dr.a0!r}; sigma = {SIG!r}; K = {K}")
    out = dict(spec='reports_CN/plan_round2c_CN.md (a716952)', h_U='mc83.py H (a = 1/2, chi = 0) via density_recursion_v1', A1={}, A2={}, A3={})
    ok = True
    for i in (UHAT, 'U-L1657'):
        a, chi = ac_of(i); Hu = dr.input_arrays(evname(i))[1]; T = T_of(a, chi); p = a**2
        dmax = float(np.abs(Hu-HU@T.T).max()); ok &= dmax <= 1e-12
        dT = float(np.linalg.det(T)); dP = float(np.prod(4*p))
        log(f"A1 {i}: a = {a.tolist()}, chi mod 2 pi = {chi.tolist()}; max over k, j of |h_k(u) - T(u) h_U,k| = {dmax!r} (<= 1e-12: {dmax <= 1e-12}); "
            f"det T = {dT!r} vs prod_m 4 p_m = {dP!r}; sum p = {float(p.sum())!r}; power term 2 sum log(4 p_m) = {2*float(np.log(4*p).sum())!r}")
        out['A1'][i] = dict(max_abs_diff=dmax, passed=bool(dmax <= 1e-12), det_T=dT, prod_4p=dP, power_term=2*float(np.log(4*p).sum()))
    log(f"A1 T(u) = blockdiag(1, 2 a_m R(chi_m)), R(chi) = [[cos chi, sin chi], [-sin chi, cos chi]]; verified to 1e-12 for {UHAT} and U-L1657: {ok}")
    if not ok:
        log("A1 FAILED: STOPPED"); sys.exit(1)
    # --- A2 -----------------------------------------------------------------------------------------------------------------------
    W = {}
    wU_fast = np.load(os.path.join(EV, 'results', 'fast', 'w_B_U_n24_m8.npy'))
    for i in IDS:
        tb = time.time(); rec, J, r = ef.evaluate_fast(evname(i), 24); w = np.asarray(r['W'], float); W[i] = w
        np.save(os.path.join(RES, f'w_{i}.npy'), w)
        Ja = np.load(os.path.join(EV, 'round2a', 'results', f'J_{i}.npy')); r7 = R7[i]
        sJ = np.array_equal(J, Ja); sS = rec['S'] == r7['S']; sN = rec['EN'] == r7['EN']; sL = rec['logdet'] == r7['logdet']
        nfail = int((~r['assert_ok']).sum())
        log(f"A2 {i}: evaluator_fast.evaluate_fast('{evname(i)}', 24) (B, n = 24, margin 8 Delta): S {rec['S']!r}, E[N] {rec['EN']!r}, log det J "
            f"{rec['logdet']!r}; w_1 = {float(w[0])!r}, sum w = {float(w.sum())!r}, w in [{float(w.min())!r}, {float(w.max())!r}]; assertion failures {nfail}; "
            f"bitwise equal to the round-2a archive: J (round2a/results/J_{i}.npy) {sJ}, S {sS}, E[N] {sN}, log det {sL} (round2a/results/recheck7.json); "
            f"{time.time()-tb:.1f} s -> results/w_{i}.npy")
        out['A2'][i] = dict(S=rec['S'], EN=rec['EN'], logdet=rec['logdet'], sum_w=float(w.sum()), J_bitwise=bool(sJ), S_bitwise=bool(sS),
                            EN_bitwise=bool(sN), logdet_bitwise=bool(sL), assertion_failures=nfail, file=f'results/w_{i}.npy')
        if not (sJ and sS and sN and sL and nfail == 0):
            log(f"A2 {i}: the recomputed weights do not reproduce the round-2a archive: STOPPED"); sys.exit(1)
    su = np.array_equal(W['UNIF'], wU_fast)
    log(f"A2 w_U recomputed vs the archived fast-path weights results/fast/w_B_U_n24_m8.npy (speed-up regression archive): bitwise equal {su}; "
        f"max |diff| {float(np.abs(W['UNIF']-wU_fast).max())!r}")
    out['A2']['UNIF']['fast_archive_bitwise'] = bool(su)
    # --- A3 -----------------------------------------------------------------------------------------------------------------------
    for idx in (1, 2):
        src = SRC[idx]; Mi = M_of(HU, W[src]); Lc, ld = factor(Mi)
        A = cho_solve((Lc, True), np.eye(NPAR)); gam = gamma_of(Lc, HU)
        sl, ld_lu = np.linalg.slogdet(Mi); ev = np.linalg.eigvalsh(Mi)
        cM = float(np.linalg.cond(Mi)); cA = float(np.linalg.cond(A)); asym = float(np.abs(A-A.T).max()/np.abs(A).max())
        gA = np.einsum('kj,jl,kl->k', HU, A, HU)/SIG**2
        gb = float(gam.mean())
        np.save(os.path.join(RES, f'M_A{idx}.npy'), Mi); np.save(os.path.join(RES, f'A{idx}.npy'), A); np.save(os.path.join(RES, f'gamma_A{idx}.npy'), gam)
        log(f"A3 A_{idx} = M(w({src}))^-1: M = sigma^-2 sum_k w_k h_U,k h_U,k^T from results/w_{src}.npy; log det M_{idx} (Cholesky) = {ld!r} "
            f"(slogdet {float(ld_lu)!r}, sign {float(sl)}); log det A_{idx} = {-ld!r}")
        log(f"A3 A_{idx}: cond_2(M_{idx}) = {cM!r}, cond_2(A_{idx}) = {cA!r}; eigenvalues of M_{idx} in [{float(ev[0])!r}, {float(ev[-1])!r}]; "
            f"A_{idx} = cho_solve(L, I) (two triangular solves, no explicit inverse), relative asymmetry {asym!r} -> results/A{idx}.npy, results/M_A{idx}.npy")
        log(f"A3 gamma^({idx})_k = sigma^-2 |L^-1 h_U,k|^2: gamma_bar = mean_k = {gb!r}; min {float(gam.min())!r} (k = {int(gam.argmin())+1}); "
            f"max {float(gam.max())!r} (k = {int(gam.argmax())+1}); gamma_1 = {float(gam[0])!r}; sum_k gamma_k = {float(gam.sum())!r}; "
            f"20 gamma_bar = {HI_MULT*gb!r}; via the stored A_{idx} (h^T A h) max relative difference {float(np.abs(gA-gam).max()/gam.max())!r} "
            f"-> results/gamma_A{idx}.npy")
        rec = dict(source=src, logdet_M=ld, logdet_M_slogdet=float(ld_lu), logdet_A=-ld, cond_M=cM, cond_A=cA, eig_M=[float(v) for v in ev],
                   gamma_bar=gb, gamma_min=float(gam.min()), gamma_max=float(gam.max()), k_min=int(gam.argmin())+1, k_max=int(gam.argmax())+1,
                   gamma_1=float(gam[0]), gamma_sum=float(gam.sum()), lam_hi=HI_MULT*gb)
        if idx == 2:
            JU = np.load(os.path.join(EV, 'round2a', 'results', 'J_UNIF.npy')); same = np.array_equal(Mi, JU)
            log(f"A3 M_2 = M(w_U) vs the archived J(u_unif) (round2a/results/J_UNIF.npy; T(u_unif) = I): bitwise equal {same}; "
                f"max |diff| / max |J| {float(np.abs(Mi-JU).max()/np.abs(JU).max())!r}")
            rec['M_equals_J_UNIF_bitwise'] = bool(same)
        out['A3'][f'A{idx}'] = rec
    json.dump(out, open(os.path.join(RES, 'part_a.json'), 'w'), indent=1)


# --- DP with the gamma-weighted reward ---------------------------------------------------------------------------------------------
def dp2d_gamma(lam, gam, model):
    """round2b mdp2d_estimate.dp2d (window half-width 1, no policy storage) with the reward of step k = gam[k-1] c - lam q.
    Returns V~_lambda = max_{l_1} sum_j phat_j V_2(j - l_1, l_1) and the maximising v_1 index."""
    t, c, q, P, phat, Ninv, N = model
    V = np.zeros((NL, G)); Vout = np.empty((NL, G)); Q = np.empty((NL, G))
    Gp = np.empty((NL+2, G)); Hs = np.empty((NL+2, G))
    for k in range(K, 1, -1):
        r = gam[k-1]*c-lam*q
        m2._q_all(V, P, q, phat, Ninv, M, LH, Q)
        m2._wmax(Q, 1, r, Vout, Gp, Hs)
        V, Vout = Vout, V
    return m2.init_value(V, phat)

def dp1d_gamma(lam, gam, model):
    """round2a mdp_estimate.dual_value on the grid of round2b/grid30.py (no policy storage) with the reward of step k
    = gam[k-1] c - lam q (V_K = reward of step K)."""
    t, c, q, P, phat, Ninv, N = model
    V = gam[K-1]*c-lam*q; Vout = np.empty(G); pol = np.zeros(G, np.int64)
    for k in range(K-1, 1, -1):
        mdp._max_step(V, gam[k-1]*c-lam*q, P, q, phat, Ninv, M, g.ORDER, Vout, pol); V, Vout = Vout, V
    Vt, l1 = mdp._init_value(V, phat, g.ORDER)
    return float(Vt), int(l1)

def forward2d_steps(codes_at, l1, model):
    """round2b mdp2d_estimate.forward2d (same transport, m2._fwd) recording E[c] and E[q] per step k = 2..K (index k)."""
    t, c, q, P, phat, Ninv, N = model
    mu = np.zeros((NL, G)); mu[l1+LH] = mdp.initial_mu(phat, l1); out = np.empty((NL, G))
    Ec = np.zeros(K+1); Eq = np.zeros(K+1); mass = np.zeros(K+1)
    for k in range(2, K+1):
        Ec[k] = float((mu@c).sum()); Eq[k] = float((mu@q).sum()); mass[k] = float(mu.sum())
        if k < K:
            m2._fwd(mu, codes_at(k), P, q, phat, Ninv, M, LH, out); mu, out = out, mu
    return Ec, Eq, mass

def backward_fixed2d_gamma(codes_at, l1, gam, model):
    """round2b mdp2d_estimate.backward_fixed2d with the reward of step k = gam[k-1] c: sum_{k>=2} gamma_k E[c] at the initial state."""
    t, c, q, P, phat, Ninv, N = model
    V = np.zeros((NL, G)); Vout = np.empty((NL, G)); Q = np.empty((NL, G)); one = np.ones((NL, G), np.int8)
    for k in range(K, 1, -1):
        m2._q_all(V, P, q, phat, Ninv, M, LH, Q)
        m2._apply_codes(Q, codes_at(k) if k < K else one, gam[k-1]*c, Vout)
        V, Vout = Vout, V
    return float(mdp._init_value(V[l1+LH], phat, np.array([l1], np.int64))[0])

def forward1d_steps(lseq, l1, model):
    """round2a mdp_estimate.forward (same transport, _forward_step) for an open-loop policy, recording E[c] per step."""
    t, c, q, P, phat, Ninv, N = model
    mu = mdp.initial_mu(phat, l1); out = np.empty(G); used = np.zeros(NL, np.bool_); Ec = np.zeros(K+1); Eq = np.zeros(K+1)
    for k in range(2, K+1):
        Ec[k] = float(mu@c); Eq[k] = float(mu@q)
        if k < K:
            pk = np.full(G, lseq[k], np.int64); used[:] = False; used[lseq[k]+LH] = True
            mdp._forward_step(mu, P, q, phat, Ninv, M, pk, used, out); mu, out = out, mu
    return Ec, Eq

def backward_fixed1d_gamma(lseq, l1, gam, model):
    """round2a mdp_estimate.backward_fixed with the reward of step k = gam[k-1] c."""
    t, c, q, P, phat, Ninv, N = model
    V = gam[K-1]*c; A_ = np.empty(G); Cl = np.empty(G+1); Sr = np.empty(G+1); tmp = np.empty(G)
    for k in range(K-1, 1, -1):
        mdp._q_action(V, P, q, phat, Ninv, M, lseq[k], A_, Cl, Sr, tmp); V = gam[k-1]*c+tmp
    return float(sum(phat[j]*V[min(max(j-l1, 0), G-1)] for j in range(G)))


# --- Part C ------------------------------------------------------------------------------------------------------------------------
def checks_passed():
    p = os.path.join(RES, 'checks.json')
    if not os.path.exists(p): return False
    return bool(json.load(open(p)).get('all_passed', False))

def stage_outer(idx, model_name, log):
    if not checks_passed():
        log("C1 refused: results/checks.json does not record that checks B1-B4 all passed (spec §3: no estimate before the checks)"); sys.exit(1)
    Mi, Lc, ldM, A, gam = load_A(idx); gb = float(gam.mean()); g1 = float(gam[0]); hi = HI_MULT*gb
    const = ldM-NPAR+g1                                                   # -log det A_i - 9 + gamma_1
    model = mdp.model()
    dp = dp2d_gamma if model_name == '2d' else dp1d_gamma
    tag = f'{model_name}_A{idx}'
    cache = g.Cache(os.path.join(LOGS, f'gold_{tag}.jsonl'))
    log(f"C1 {tag}: model {'2-D acceleration-constrained (Delta j in {-1, 0, 1})' if model_name == '2d' else '1-D unconstrained'}, grid h = sigma/30, "
        f"L = 137; A_{idx} = M(w({SRC[idx]}))^-1; -log det A_{idx} - 9 + gamma_1 = {ldM!r} - 9 + {g1!r} = {const!r}; gamma_bar = {gb!r}; "
        f"lambda in [0, 20 gamma_bar] = [0, {hi!r}]; golden section to bracket width eps_lambda = {EPS!r}; numba threads {get_num_threads()}")
    def Vlam(lam):
        r = cache.get(lam)
        if r is None:
            ts = time.time(); Vt, l1 = dp(lam, gam, model); dt_ = time.time()-ts
            r = dict(V=float(Vt), l1=int(l1), time_s=dt_, threads=get_num_threads(), load=g.loadavg()); cache.put(lam, r)
            how = f"{dt_:.1f} s; numba threads {r['threads']}; load {r['load']}"
        else:
            how = f"cached from logs/gold_{tag}.jsonl, computed in {r['time_s']:.1f} s"
        log(f"GOLD-{tag} lambda {lam!r} (lambda/gamma_bar {lam/gb!r}) V~ {r['V']!r} D {const+r['V']+BM1*lam!r} (v_1 = {r['l1']} h; {how})")
        return r['V'], r['l1']
    tg = time.time()
    evals, lam_s, Fmin, lo, hi_ = g.golden(Vlam, EPS, log, lo=0.0, hi=hi)
    Dmin = const+Fmin; tsum = sum(cache.get(k)['time_s'] for k in evals); band = SLOPE*EPS
    edge = min(lam_s, hi-lam_s) <= EPS
    log(f"C1 {tag} final bracket [{lo!r}, {hi_!r}] (width {hi_-lo!r}); {len(evals)} evaluations; wall of this call {time.time()-tg:.1f} s; "
        f"sum of per-lambda DP times {tsum:.1f} s ({tsum/3600:.4f} h)")
    log(f"C1 {tag} 估算 min_lambda D(A_{idx}, lambda) = {Dmin!r} at lambda* = {lam_s!r} (lambda*/gamma_bar = {lam_s/gb!r}; V~ = {evals[lam_s][0]!r}; "
        f"v_1 = {evals[lam_s][1]} h); lambda* within eps_lambda of an end of [0, 20 gamma_bar]: {edge}; outer-error band 4684 eps_lambda = {band!r} nat "
        f"(<= 0.02: {band <= 0.02}); min D - {LD_JC} = {Dmin-LD_JC!r} nat (Delta m_D scale; an estimate on the discrete model)")
    res = dict(case=tag, model=model_name, A=idx, A_source=SRC[idx], logdet_A=-ldM, gamma_1=g1, gamma_bar=gb, const=const, lam_range=[0.0, hi],
               eps_lambda=EPS, lam_star=lam_s, lam_star_repr=repr(lam_s), lam_star_over_gamma_bar=lam_s/gb, V_at_lam_star=evals[lam_s][0],
               v1=evals[lam_s][1], D_min=Dmin, D_min_minus_LDJC=Dmin-LD_JC, LD_JC=LD_JC, band=band, lam_star_at_edge=bool(edge),
               n_evals=len(evals), bracket=[lo, hi_], dp_time_sum_s=tsum, wall_s=time.time()-tg,
               evals={repr(k): dict(V=v[0], D=const+v[0]+BM1*k, l1=v[1]) for k, v in sorted(evals.items())})
    json.dump(res, open(os.path.join(RES, f'outer_{tag}.json'), 'w'), indent=1)
    return res

def stage_sroute(log):
    import b5_conversion as b5                                             # round2a/b5_conversion.py (on sys.path via grid30)
    ns = b5.bound_namespace()
    ns['exchange'](ns['H_u'][::25], 50.0, 1.0, tol=1e-3, max_sweeps=5)    # numba compile, as b5_conversion
    log(f"C2 route-1 solver: round2a/b5_conversion.solve = bound/bound_common.exchange(H_u, W, cap = 1, tol = 1e-10, batch = 256) + "
        f"certificate_full (unchanged); H_u equals mc83 H (h_U of Part A) bitwise: {np.array_equal(ns['H_u'], h_uniform())}; "
        f"bound_common LD_SOD = {ns['LD_SOD']!r} (log det J_C, rounded {LD_JC} in the spec)")
    out = {}
    for lab, W in (('reproduction of round2a/logs/b5_conversion.log line 6 (W = S~_dual of round 2a, U_cp 206.97779417261597)', 1210.9563921988488),
                   ('S-route value at S~_acc of round 2b', S_2B)):
        cx, rx, tt = b5.solve(ns, W)
        log(f"C2 U at W = {W!r} ({lab}): F(xi~) = {cx['F']!r}; certificate value U_cp (cutting plane) = {cx['U_cp']!r}; solver gap {cx['gap_cp']!r}; "
            f"duality value U_dual {float(cx['U_dual'])!r}; {rx['sweeps']} sweeps, {tt:.1f} s")
        out[repr(W)] = dict(label=lab, W=W, F=cx['F'], U_cp=cx['U_cp'], gap_cp=cx['gap_cp'], U_dual=float(cx['U_dual']), sweeps=rx['sweeps'], time_s=tt)
    r = out[repr(S_2B)]
    log(f"C2 reproduction difference U_cp(1210.9563921988488) - 206.97779417261597 = {out[repr(1210.9563921988488)]['U_cp']-206.97779417261597!r}")
    log(f"C2 估算 S-route value U(619.1292140245644; u_unif) - {LD_JC} = {r['U_cp']!r} - {LD_JC} = {r['U_cp']-LD_JC!r} nat (certificate value U_cp, "
        f"cap 1; solver gap {r['gap_cp']!r}; an estimate on the discrete model: S~_acc is a discrete-model estimate)")
    res = dict(W=S_2B, U_cp=r['U_cp'], F=r['F'], gap=r['gap_cp'], U_dual=r['U_dual'], value=r['U_cp']-LD_JC, LD_JC=LD_JC, runs=out)
    json.dump(res, open(os.path.join(RES, 'part_c2.json'), 'w'), indent=1)
    return res

def stage_table(log):
    rows = []
    for mod, mlab in (('2d', '二维加速度约束'), ('1d', '一维无约束')):
        for idx in (1, 2):
            r = json.load(open(os.path.join(RES, f'outer_{mod}_A{idx}.json')))
            rows.append(dict(route='direct', case=r['case'], label=f'直接路线，A_{idx}，{mlab}', value=r['D_min_minus_LDJC'], D_min=r['D_min'],
                             lam=r['lam_star'], lam_over_gbar=r['lam_star_over_gamma_bar'], band=r['band'], n=r['n_evals'], t=r['dp_time_sum_s']))
    s = json.load(open(os.path.join(RES, 'part_c2.json')))
    rows.append(dict(route='S', case='S-route', label='S 路线，U(619.1292140245644; u_unif)，cap 1', value=s['value'], D_min=s['U_cp'], gap=s['gap']))
    log("C3 table (估算, nat on the Delta m_D scale: value - 197.197):")
    log("C3 | case | value - 197.197 | value | lambda* | lambda*/gamma_bar | band 4684 eps | evaluations | DP time (s) |")
    for r in rows:
        if r['route'] == 'direct':
            log(f"C3 | {r['case']} | {r['value']!r} | {r['D_min']!r} | {r['lam']!r} | {r['lam_over_gbar']!r} | {r['band']!r} | {r['n']} | {r['t']:.1f} |")
        else:
            log(f"C3 | {r['case']} | {r['value']!r} | {r['D_min']!r} | — | — | solver gap {r['gap']!r} | — | — |")
    sv = s['value']
    for r in rows[:4]: log(f"C3 {r['case']} - S-route = {r['value']-sv!r} nat")
    d = {r['case']: r['value'] for r in rows}
    log(f"C3 A_2 - A_1 (2-D) = {d['2d_A2']-d['2d_A1']!r}; (1-D) = {d['1d_A2']-d['1d_A1']!r}; 1-D - 2-D (A_1) = {d['1d_A1']-d['2d_A1']!r}; (A_2) = {d['1d_A2']-d['2d_A2']!r}")
    json.dump(dict(rows=rows), open(os.path.join(RES, 'part_c3.json'), 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['construct', 'outer', 'sroute', 'table'])
    ap.add_argument('--A', type=int, choices=[1, 2]); ap.add_argument('--model', choices=['2d', '1d']); ap.add_argument('--threads', type=int, default=None)
    args = ap.parse_args()
    if args.threads: set_num_threads(args.threads)
    logname = {'construct': 'part_a.log', 'outer': 'part_c1.log', 'sroute': 'part_c2.log', 'table': 'part_c3.log'}[args.stage]
    sys.stdout = dr.Tee(os.path.join(LOGS, logname), 'a')
    def log(m): print(m, flush=True)
    T0 = time.time()
    log(f"\n=== logdet_dual.py {args.stage}{f' --A {args.A} --model {args.model}' if args.stage == 'outer' else ''} started "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} (round 2c; 估算 on the discrete model); numba threads {get_num_threads()}; load {g.loadavg()}")
    if args.stage == 'construct': stage_construct(log)
    elif args.stage == 'outer': stage_outer(args.A, args.model, log)
    elif args.stage == 'sroute': stage_sroute(log)
    else: stage_table(log)
    log(f"=== logdet_dual.py {args.stage} finished {time.strftime('%Y-%m-%d %H:%M:%S')}; wall {time.time()-T0:.1f} s")
