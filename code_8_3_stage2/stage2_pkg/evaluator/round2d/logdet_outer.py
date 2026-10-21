"""Round 2d: outer minimisation over A of the direct log-det dual (spec reports_CN/plan_round2d_CN.md, frozen, commit ac058c4).

Everything this file produces is an estimate on the discrete model (估算); nothing is certified and no number is called a bound.

Whitened parametrisation (spec §1.1). M_0 = M(w(u_hat)), u_hat = P5514 (round2c/results/M_A1.npy, A_1 = M_0^-1 of round 2c), lower
Cholesky factor M_0 = L L^T (round2c factor()); g_k = sigma^-1 L^-1 h_U,k (one triangular solve, no inverse is formed). A = L^-T A~ L^-1,
so gamma_k(A~) = sigma^-2 h_U,k^T A h_U,k = g_k^T A~ g_k and -log det A = -log det A~ + log det M_0. L^-1 stands for the spec's M_0^-1/2:
L^-1 = Q M_0^-1/2 with Q orthogonal, so the two differ by the orthogonal change of variables A~ -> Q^T A~ Q, which leaves D, the Frobenius
trust region and ||A~ - I||_F unchanged (DECISION A-1). A~ = I is round 2c's A_1.
    D(A~, lambda) = log det M_0 - log det A~ - 9 + gamma_1(A~) + lambda (B - 1) + V~_{A~,lambda},   B - 1 = 315.892.
Oracle (Part A2): round 2c's gamma-weighted 2-D DP (logdet_dual.dp2d_gamma: same kernels, same order) with the policy stored in RAM by
round 2b's _codes/_pack, then round 2c's forward pass (logdet_dual.forward2d_steps) under that policy: w_k = E_pi[c(delta_k)] (k >= 2),
w_1 = 1, n = sum_{k>=2} E_pi[q(delta_k)]. Subgradient G_A~ = -A~^-1 + sum_{k>=1} w_k g_k g_k^T, G_lambda = (B - 1) - n; the cut matrix is
M~^(2+) = sum_{k>=2} w_k g_k g_k^T (gamma_1 is exact in the master objective; the cuts model V~ only).
Outer loop (Part C, spec §2): cutting planes with the exact -log det A~ + gamma_1 in the master objective, Frobenius trust region on A~
around the centre (last accepted point), lambda in [0, 0.5] (not in the trust region, as the spec writes it); master problem in cvxpy,
SCS first and Clarabel when SCS does not return 'optimal' (DECISION C-2); stop at D_best - m_global <= 0.02 nat or after 300 oracle
calls. Checkpoint after every iteration (results/cuts.npz + results/state.json, atomic replace); a killed run resumes from it.

Stages (from this directory; each appends to its own log in logs/):
  outer [--threads 4]   Part C; refuses to run unless results/checks.json records B1 and B2 passed -> logs/outer.log (one line per
                        iteration), logs/oracle.log (oracle details), results/cuts.npz, results/state.json, results/w_iter/
  final                 C5 at the best evaluated point -> logs/final.log, results/final.json, results/A_star.npy, results/w_star.npy
  table                 Part D comparison row -> logs/part_d.log, results/part_d.json
  plot                  fig_convergence.png from results/
Usage: setsid nohup python3 logdet_outer.py outer --threads 4 > logs/outer.stdout 2>&1 &"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EV, 'round2c'))
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
import logdet_dual as ld
from logdet_dual import g, mdp, m2, dr, K, NL, LH, G, M, NPAR, BM1, LD_JC, SIG
from numba import set_num_threads, get_num_threads

R2C = os.path.join(EV, 'round2c')
OUT2C = json.load(open(os.path.join(R2C, 'results', 'outer_2d_A1.json')))
LAM_2C = OUT2C['lam_star']                  # 0.020120725733454057, round2c/logs/part_c1_2d_A1.log:31 (archived full precision)
DM_2C = OUT2C['D_min_minus_LDJC']           # 4.385285228783914 (2-D, A_1, fixed A)
V_2C = OUT2C['V_at_lam_star']               # 6.2948236330331815
S_ROUTE = json.load(open(os.path.join(R2C, 'results', 'part_c2.json')))['value']    # 3.7756005385593596
LAM_LO, LAM_HI = 0.0, 0.5                   # lambda in [0, 0.5] (spec §2, task C1)
R0, R_MAX, R_MIN = 0.5, 2.0, 1e-3           # trust radius: start, cap, floor (spec §2)
ETA, GROW, SHRINK = 0.25, 1.5, 0.5          # ratio threshold, r <- min(1.5 r, 2) on accept, r <- max(r/2, 1e-3) on reject
GAP_STOP, IT_MAX = 0.02, 300                # stop when D_best - m_global <= 0.02 nat, or after 300 oracle calls
SCS_OPTS = dict(eps_abs=1e-8, eps_rel=1e-8, max_iters=100000)    # DECISION C-2
LOGS = os.path.join(HERE, 'logs'); RES = os.path.join(HERE, 'results'); WIT = os.path.join(RES, 'w_iter')
for d_ in (LOGS, RES, WIT): os.makedirs(d_, exist_ok=True)
CUTS = os.path.join(RES, 'cuts.npz'); STATE = os.path.join(RES, 'state.json')


# --- Part A1: whitened objective ---------------------------------------------------------------------------------------------------
M0 = np.load(os.path.join(R2C, 'results', 'M_A1.npy'))
L0, LDM0 = ld.factor(M0)                    # M_0 = L0 L0^T; log det M_0 = 2 sum log diag L0 (= -log det A_1 of round 2c)
HU = ld.h_uniform()
VW = solve_triangular(L0, HU.T, lower=True) # 9 x K; column p = L0^-1 h_U,k (k = p + 1); g_k = VW[:, p] / sigma
G1 = VW[:, 0]/SIG                           # g_1
C1 = np.outer(VW[:, 0], VW[:, 0])/SIG**2    # g_1 g_1^T
EYE = np.eye(NPAR)

def gamma_of(At):
    """gamma_k(A~) = g_k^T A~ g_k = sigma^-2 (L0^-1 h)^T A~ (L0^-1 h), all k. A~ VW is put in VW's (Fortran) layout so that at A~ = I
    the einsum is round 2c's gamma_of bit for bit (I VW = VW exactly)."""
    return np.einsum('ij,ij->j', VW, np.asfortranarray(At@VW))/SIG**2

def gram(w, k0):
    """sum_{k >= k0} w_k g_k g_k^T (w indexed by p = k - 1)."""
    V = VW[:, k0-1:]
    return (V*w[k0-1:])@V.T/SIG**2

def logdet_pd(At):
    Lt = cholesky(At, lower=True)
    return Lt, 2.0*float(np.log(np.diag(Lt)).sum())

def D_const(At):
    """log det M_0 - log det A~ - 9 + gamma_1(A~), the part of D outside lambda (B - 1) + V~ (same order as round 2c's const)."""
    Lt, ldA = logdet_pd(At)
    return LDM0-ldA-NPAR+float(gamma_of(At)[0]), ldA, Lt

def unwhiten(At):
    """A = L0^-T A~ L0^-1 (two triangular solves)."""
    Y = solve_triangular(L0, At, lower=True, trans='T')
    return solve_triangular(L0, Y.T, lower=True, trans='T').T


# --- Part A2: oracle ----------------------------------------------------------------------------------------------------------------
NB = m2.NB
_POL = {}

def policy_buffer():
    if 'p' not in _POL: _POL['p'] = np.empty((K-1, NB), np.uint8)      # packed policy, row k - 2 (k = 2..K), 1.15 GiB, in RAM
    return _POL['p']

def dp2d_gamma_store(lam, gam, model, pol):
    """round2c logdet_dual.dp2d_gamma (reward of step k = gam[k-1] c - lam q, window half-width 1) with the A = 1 policy stored by
    round 2b's _codes (tie rule of round 2a) and _pack into pol. Returns V~, v_1 index and the number of states where the chosen
    action's Q + r differs from the window maximum (must be 0)."""
    t, c, q, P, phat, Ninv, N = model
    V = np.zeros((NL, G)); Vout = np.empty((NL, G)); Q = np.empty((NL, G))
    Gp = np.empty((NL+2, G)); Hs = np.empty((NL+2, G)); codes = np.empty((NL, G), np.int8); bad = 0
    for k in range(K, 1, -1):
        r = gam[k-1]*c-lam*q
        m2._q_all(V, P, q, phat, Ninv, M, LH, Q)
        m2._wmax(Q, 1, r, Vout, Gp, Hs)
        bad += m2._codes(Q, Vout, r, LH, codes); m2._pack(codes.reshape(-1), pol[k-2])
        V, Vout = Vout, V
    Vt, l1 = m2.init_value(V, phat)
    return Vt, l1, bad

def D_value(At, lam, model):
    """D(A~, lambda) from round 2c's DP without policy storage (Part B checks): returns D, V~, v_1, DP time."""
    const, ldA, _ = D_const(At); gam = gamma_of(At)
    tb = time.time(); Vt, l1 = ld.dp2d_gamma(lam, gam, model); tdp = time.time()-tb
    return const+Vt+BM1*lam, Vt, l1, tdp

def oracle(At, lam, model):
    """Part A2: D, V~, policy forward (w, n), subgradient (G_A~, G_lambda), cut matrix M~^(2+) at (A~, lambda)."""
    pol = policy_buffer(); cf = np.empty(NL*G, np.int8)
    const, ldA, Lt = D_const(At); gam = gamma_of(At)
    tb = time.time(); Vt, l1, bad = dp2d_gamma_store(lam, gam, model, pol); tdp = time.time()-tb
    def at(k):
        m2._unpack(pol[k-2], cf); return cf.reshape(NL, G)
    tb = time.time(); Ec, Eq, mass = ld.forward2d_steps(at, l1, model); tfw = time.time()-tb
    w = np.empty(K); w[0] = 1.0; w[1:] = Ec[2:K+1]; eq = np.zeros(K); eq[1:] = Eq[2:K+1]
    n = float(eq[1:].sum()); fw = float(gam[1:]@w[1:])-lam*n
    D = const+Vt+BM1*lam
    Mt2 = gram(w, 2); GA = -cho_solve((Lt, True), EYE)+C1+Mt2; GA = 0.5*(GA+GA.T); Mt2 = 0.5*(Mt2+Mt2.T)
    return dict(lam=lam, D=D, V=Vt, l1=int(l1), bad=int(bad), const=const, logdetA=ldA, gamma1=float(gam[0]), w=w, Eq=eq, n=n, Mt2=Mt2, GA=GA,
                Glam=BM1-n, fwd_value=fw, fwd_rel=abs(fw-Vt)/abs(Vt), mass_K=float(mass[K]), t_dp=tdp, t_fwd=tfw)


# --- Part C: master problem and cut model ------------------------------------------------------------------------------------------
def cut_offsets(C):
    """a_i = V~_i - <M~_i, A~_i> + lambda_i n_i, so that cut_i(A~, lambda) = a_i + <M~_i, A~> - n_i lambda."""
    return C['V']-np.einsum('mij,mij->m', C['Mt2'], C['X'])+C['lam']*C['n']

def model_value(C, X, lam, upto=None):
    """analytic cutting-plane model at (X, lam) with the cuts 0..upto-1: log det M_0 - log det X - 9 + g_1^T X g_1 + lam (B - 1)
    + max_i cut_i(X, lam). Returns the model value and the cut values."""
    m = len(C['V']) if upto is None else upto
    cuts = cut_offsets({k: C[k][:m] for k in ('V', 'Mt2', 'X', 'lam', 'n')})+np.einsum('mij,ij->m', C['Mt2'][:m], X)-C['n'][:m]*lam
    const, _, _ = D_const(X)
    return const+BM1*lam+float(cuts.max()), cuts

def cut_validity(C):
    """spec §3 item 3: for every evaluated i, model(A~_i, lambda_i) - D_i = max_l cut_l(A~_i, lambda_i) - V~_i (>= 0 because l = i
    gives V~_i). Returns the largest value of cut_l(x_i) - V~_i over l != i (> 0 is a violation of D_i >= model) and its (i, l)."""
    m = len(C['V']); a = cut_offsets(C)
    T = a[None, :]+np.einsum('lij,mij->ml', C['Mt2'], C['X'])-C['lam'][:, None]*C['n'][None, :]     # T[i, l] = cut_l(x_i)
    E = T-C['V'][:, None]
    if m == 1: return -np.inf, (0, 0), float(abs(E[0, 0]))
    np.fill_diagonal(E, -np.inf); i, l = np.unravel_index(int(np.argmax(E)), E.shape)
    selfdiff = float(np.abs(np.diag(T)-C['V']).max())
    return float(E[i, l]), (int(i), int(l)), selfdiff

def master(C, Xc, r, use_tr):
    """the master problem of spec §2 (constants log det M_0 - 9 added after the solve). Returns the minimiser, the solver's value,
    the analytic model value at the minimiser and the solver record (SCS first; Clarabel when SCS is not 'optimal')."""
    import cvxpy as cp
    m = len(C['V']); a = cut_offsets(C); Mm = C['Mt2'].reshape(m, -1, order='F')
    X = cp.Variable((NPAR, NPAR), PSD=True); lam = cp.Variable(); t = cp.Variable()
    cons = [t >= a+Mm@cp.vec(X, order='F')-lam*C['n'], lam >= LAM_LO, lam <= LAM_HI]
    if use_tr: cons.append(cp.norm(X-Xc, 'fro') <= r)
    prob = cp.Problem(cp.Minimize(-cp.log_det(X)+G1@X@G1+BM1*lam+t), cons)
    rec = dict(tries=[])
    for name, opts in (('SCS', SCS_OPTS), ('CLARABEL', {})):
        tb = time.time()
        try:
            prob.solve(solver=name, **opts); st = prob.status
            if st in ('optimal', 'optimal_inaccurate'):
                try: cholesky(0.5*(X.value+X.value.T), lower=True)
                except np.linalg.LinAlgError: st = f'{st} but A~ not positive definite'
        except Exception as e:                                             # solver error: logged, next solver
            st = f'error: {type(e).__name__}: {str(e)[:120]}'
        rec['tries'].append(dict(solver=name, status=st, time_s=time.time()-tb,
                                 value=(float(prob.value) if st in ('optimal', 'optimal_inaccurate') else None)))
        if st == 'optimal': break
    if rec['tries'][-1]['status'] not in ('optimal', 'optimal_inaccurate'):
        raise RuntimeError(f'master problem not solved: {rec}')
    rec['solver'] = rec['tries'][-1]['solver']; rec['status'] = rec['tries'][-1]['status']
    Xv = 0.5*(X.value+X.value.T); lv = float(lam.value); lc = min(max(lv, LAM_LO), LAM_HI)
    val = LDM0-NPAR+float(prob.value)
    mv, _ = model_value(C, Xv, lc)
    rec.update(lam_clip=lc-lv, tr_norm=(float(np.linalg.norm(Xv-Xc)) if Xc is not None else None), r=r)
    return Xv, lc, val, mv, rec


# --- checkpoint ------------------------------------------------------------------------------------------------------------------
CUT_KEYS = ('X', 'lam', 'V', 'D', 'Mt2', 'n', 'l1', 'logdetA', 'gamma1', 'fwd_rel', 'mass_K', 'bad', 't_dp', 't_fwd')
HIST_KEYS = ('m_pred', 'ratio', 'accepted', 'r', 'm_global', 'm_global_at', 'gap', 'D_best', 'viol', 'center', 't_master', 't_iter')

def save_state(C, H, S):
    np.savez(CUTS+'.tmp.npz', **{k: np.asarray(C[k]) for k in CUT_KEYS}, **{f'h_{k}': np.asarray(H[k], float) for k in HIST_KEYS},
             Xnext=S['Xnext'], Xc=S['Xc'])
    os.replace(CUTS+'.tmp.npz', CUTS)
    js = {k: v for k, v in S.items() if k not in ('Xnext', 'Xc')}
    with open(STATE+'.tmp', 'w') as f: json.dump(js, f, indent=1)
    os.replace(STATE+'.tmp', STATE)

def load_state():
    z = np.load(CUTS); S = json.load(open(STATE))
    C = {k: list(z[k]) for k in CUT_KEYS}; H = {k: list(z[f'h_{k}']) for k in HIST_KEYS}
    S['Xnext'] = z['Xnext']; S['Xc'] = z['Xc']
    return C, H, S

def as_arrays(C):
    return {k: np.asarray(v) for k, v in C.items()}

def checks_passed():
    p = os.path.join(RES, 'checks.json')
    return os.path.exists(p) and bool(json.load(open(p)).get('all_passed', False))


def stage_outer(log, olog):
    if not checks_passed():
        log("C refused: results/checks.json does not record that checks B1 and B2 passed (task: checks must pass before the outer loop)")
        sys.exit(1)
    model = mdp.model()
    if os.path.exists(CUTS) and os.path.exists(STATE):
        C, H, S = load_state(); j0 = len(C['V'])
        log(f"C resume from results/cuts.npz + results/state.json: {j0} cuts, centre = iteration {S['center']}, r = {S['r']!r}, "
            f"next point lambda = {S['lam_next']!r}, m_next = {S['m_next']!r}; status {S['status']}")
        if S['status'] != 'running':
            log("C nothing to do: the run has already stopped"); return
    else:
        C = {k: [] for k in CUT_KEYS}; H = {k: [] for k in HIST_KEYS}; j0 = 0
        S = dict(status='running', center=0, r=R0, Dc=None, best=0, lam_next=LAM_2C, m_next=None, Xnext=EYE.copy(), Xc=EYE.copy(),
                 t_start=time.strftime('%Y-%m-%d %H:%M:%S'), solver_log=[])
        log(f"C start (A~_0, lambda_0) = (I, lambda*_2c = {LAM_2C!r}); trust radius r = {R0}; lambda in [{LAM_LO}, {LAM_HI}]; "
            f"stop at D_best - m_global <= {GAP_STOP} nat or {IT_MAX} oracle calls; SCS {SCS_OPTS}, Clarabel defaults when SCS is not optimal")
        log("C columns: IT j | D_j - 197.197 | lambda_j | ||A~_j - I||_F | n_j | ratio | accept/reject | r (after update) | centre | "
            "D_best - 197.197 | m_global - 197.197 | gap | m_{j+1} - 197.197 | master TR: solver status | master no-TR: solver status | "
            "max cut violation | time oracle / master / iteration (s)")
    for j in range(j0, IT_MAX):
        ti = time.time()
        X = np.asarray(S['Xnext']); lam = float(S['lam_next'])
        o = oracle(X, lam, model)
        np.savez(os.path.join(WIT, f'it_{j:03d}.npz'), w=o['w'], Eq=o['Eq'])
        olog(f"ORACLE {j}: lambda {lam!r}; log det A~ {o['logdetA']!r}; gamma_1 {o['gamma1']!r}; V~ {o['V']!r} (v_1 = {o['l1']} h); D {o['D']!r}; "
             f"n {o['n']!r}; G_lambda {o['Glam']!r}; ||G_A~||_F {float(np.linalg.norm(o['GA']))!r}; forward value {o['fwd_value']!r} "
             f"(relative to V~ {o['fwd_rel']!r}); mass at K {o['mass_K']!r}; policy mismatches {o['bad']}; DP {o['t_dp']:.1f} s, forward "
             f"{o['t_fwd']:.1f} s; numba threads {get_num_threads()}; load {g.loadavg()}")
        for k in CUT_KEYS:
            C[k].append(X if k == 'X' else o[k] if k in o else None)
        # trust-region update for the point just evaluated (spec §2 step 4)
        if j == 0:
            ratio, acc = float('nan'), 1.0; S['Dc'] = o['D']; S['center'] = 0; S['Xc'] = X.copy()
        else:
            pred = S['Dc']-S['m_next']; actual = S['Dc']-o['D']
            ratio = actual/pred if pred > 0 else float('-inf')
            if ratio >= ETA:
                acc = 1.0; S['center'] = j; S['Dc'] = o['D']; S['Xc'] = X.copy(); S['r'] = min(GROW*S['r'], R_MAX)
            else:
                acc = 0.0; S['r'] = max(SHRINK*S['r'], R_MIN)
        Ca = as_arrays(C)
        Dall = Ca['D']; S['best'] = int(np.argmin(Dall)); Dbest = float(Dall[S['best']])
        # master without the trust region (m_global) and with it (next point)
        tm = time.time()
        Xg, lg, mg, mg_at, rec_g = master(Ca, None, None, False)
        gap = Dbest-mg
        viol, (vi, vl), selfd = cut_validity(Ca)
        stop = gap <= GAP_STOP
        if not stop and j+1 < IT_MAX:
            Xn, ln, mn_solver, mn, rec_t = master(Ca, np.asarray(S['Xc']), S['r'], True)
        else:
            Xn, ln, mn_solver, mn, rec_t = X, lam, float('nan'), float('nan'), dict(solver='—', status='not solved (stopped)', tries=[])
        t_master = time.time()-tm
        for k, v in (('m_pred', S['m_next'] if S['m_next'] is not None else float('nan')), ('ratio', ratio), ('accepted', acc), ('r', S['r']),
                     ('m_global', mg), ('m_global_at', mg_at), ('gap', gap), ('D_best', Dbest), ('viol', viol), ('center', S['center']),
                     ('t_master', t_master), ('t_iter', time.time()-ti)):
            H[k].append(v)
        S['solver_log'].append(dict(j=j, no_tr=rec_g, tr=rec_t))
        S['Xnext'] = Xn; S['lam_next'] = ln; S['m_next'] = mn; S['m_next_solver'] = mn_solver
        S['status'] = 'stopped: gap' if stop else ('stopped: iteration limit' if j+1 >= IT_MAX else 'running')
        save_state(C, H, S)
        dec = 'start' if j == 0 else ('accept' if acc else 'reject')
        sg = f"{rec_g['solver']} {rec_g['status']}" + (f" (SCS {rec_g['tries'][0]['status']})" if rec_g['solver'] != 'SCS' else '')
        st = f"{rec_t['solver']} {rec_t['status']}" + (f" (SCS {rec_t['tries'][0]['status']})" if rec_t.get('tries') and rec_t['solver'] != 'SCS' else '')
        log(f"IT {j} | {o['D']-LD_JC!r} | {lam!r} | {float(np.linalg.norm(X-EYE))!r} | {o['n']!r} | {ratio!r} | {dec} | {S['r']!r} | {S['center']} | "
            f"{Dbest-LD_JC!r} | {mg-LD_JC!r} | {gap!r} | {mn-LD_JC!r} | {st} | {sg} | {viol!r} (i = {vi}, l = {vl}; self {selfd!r}) | "
            f"{o['t_dp']+o['t_fwd']:.1f} / {t_master:.1f} / {time.time()-ti:.1f}")
        olog(f"MASTER {j}: no-TR value {mg!r} (model at its minimiser {mg_at!r}, difference {mg_at-mg!r}; lambda {lg!r}; ||A~ - I||_F "
             f"{float(np.linalg.norm(Xg-EYE))!r}; tries {rec_g['tries']}); TR value {mn_solver!r} (model at the next point {mn!r}; tries "
             f"{rec_t.get('tries')}; ||A~_next - A~_c||_F {rec_t.get('tr_norm')!r} vs r {S['r']!r}; lambda clip {rec_t.get('lam_clip')!r})")
        if S['status'] != 'running':
            log(f"C stopped after {j+1} oracle calls: {S['status']}; final gap D_best - m_global = {gap!r} nat (<= {GAP_STOP}: {gap <= GAP_STOP})")
            break


def primal_mixture(C, lam_cap=BM1):
    """diagnostic (not a spec item): max over mixtures theta of the evaluated policies of log det M(sum_i theta_i w_i) subject to
    sum_i theta_i n_i <= B - 1 (randomised closed-loop policies, spec §1.3). Whitened: M(w_i) = L0 (g_1 g_1^T + M~_i^(2+)) L0^T, so
    log det M = log det M_0 + log det(sum_i theta_i (C1 + M~_i)). Clarabel; returns the value, theta and the mixture's n."""
    import cvxpy as cp
    m = len(C['V']); th = cp.Variable(m, nonneg=True)
    Mx = sum(th[i]*(C1+C['Mt2'][i]) for i in range(m))
    prob = cp.Problem(cp.Maximize(cp.log_det(0.5*(Mx+Mx.T))), [cp.sum(th) == 1, C['n']@th <= lam_cap])
    prob.solve(solver='CLARABEL')
    tv = np.clip(th.value, 0, None); tv /= tv.sum(); Mm = C1+np.einsum('m,mij->ij', tv, C['Mt2'])
    return LDM0+logdet_pd(0.5*(Mm+Mm.T))[1], tv, float(C['n']@tv), prob.status

def stage_final(log):
    z = np.load(CUTS); S = json.load(open(STATE)); Ca = {k: z[k] for k in CUT_KEYS}; H = {k: z[f'h_{k}'] for k in HIST_KEYS}
    m = len(Ca['V']); b = int(np.argmin(Ca['D'])); X = Ca['X'][b]; lam = float(Ca['lam'][b]); D = float(Ca['D'][b]); n = float(Ca['n'][b])
    wz = np.load(os.path.join(WIT, f'it_{b:03d}.npz')); w = wz['w']; nchk = float(wz['Eq'][1:].sum())
    nrm = float(np.linalg.norm(X-EYE)); ev = np.linalg.eigvalsh(X); okn = abs(n-BM1) <= 5
    log(f"C5 status {S['status']}; {m} oracle calls; best evaluated point = iteration {b} (centre at stop: iteration {S['center']}, r {S['r']!r})")
    log(f"C5 估算 min D - 197.197 = D_best - 197.197 = {D-LD_JC!r} nat (D_best = {D!r}); final gap D_best - m_global = {float(H['gap'][-1])!r} nat "
        f"(<= {GAP_STOP}: {float(H['gap'][-1]) <= GAP_STOP}); m_global - 197.197 = {float(H['m_global'][-1])-LD_JC!r}")
    log(f"C5 ||A~* - I||_F = {nrm!r}; eigenvalues of A~* in [{float(ev[0])!r}, {float(ev[-1])!r}]; log det A~* = {float(Ca['logdetA'][b])!r}; "
        f"lambda* = {lam!r}; lambda*/lambda*_2c = {lam/LAM_2C!r} (lambda*_2c = {LAM_2C!r})")
    log(f"C5 forward of the policy at the best point: n* = sum_(k>=2) E[q] = {n!r} (from the stored per-step E[q]: {nchk!r}); n* - (B - 1) = {n-BM1!r}; "
        f"|n* - (B - 1)| <= 5: {okn}; sum_(k>=2) w*_k = {float(w[1:].sum())!r}; v_1 = {int(Ca['l1'][b])} h")
    dl = np.diff(H['m_global']); mono = float(dl.min()) if dl.size else 0.0
    vmax = float(np.max(H['viol'][np.isfinite(H['viol'])])) if np.isfinite(H['viol']).any() else float('-inf')
    log(f"C4 largest cut violation max_(i, l != i) [cut_l(x_i) - V~_i] over all iterations = {vmax!r} (at the last iteration, all pairs: "
        f"{float(H['viol'][-1])!r}; > 0 would mean D_i < model(x_i)); m_global non-decreasing: smallest step m_global_(j+1) - m_global_j = {mono!r}")
    log("C4 m_global - 197.197 sequence: " + ', '.join(f'{v-LD_JC:.6f}' for v in H['m_global']))
    acc = H['accepted'][1:]; sl = S['solver_log']
    cnt = {}
    for e in sl:
        for kk in ('no_tr', 'tr'):
            key = f"{kk}:{e[kk]['solver']}:{e[kk]['status']}"; cnt[key] = cnt.get(key, 0)+1
    ttot = float(H['t_iter'].sum()); tor = float((Ca['t_dp']+Ca['t_fwd']).sum()); tma = float(H['t_master'].sum())
    log(f"C runtime: sum over iterations {ttot:.1f} s ({ttot/3600:.3f} h; oracle {tor:.1f} s, master problems {tma:.1f} s); accepted steps "
        f"{int(acc.sum())}, rejected {int((1-acc).sum())}; master solves by solver and status {cnt}")
    log(f"C oracle diagnostics: largest forward/backward relative difference {float(Ca['fwd_rel'].max())!r}; policy mismatches total {int(Ca['bad'].sum())}")
    pv, th, nmix, pst = primal_mixture(Ca)
    supp = [(int(i), float(th[i])) for i in np.argsort(-th)[:5] if th[i] > 1e-6]
    Mfull = C1+Ca['Mt2'][b]; ldw = LDM0+logdet_pd(0.5*(Mfull+Mfull.T))[1]; trw = float(np.sum(X*Mfull))
    log(f"C5 diagnostic (not a spec item): log det M(w*) of the best point's own policy - 197.197 = {ldw-LD_JC!r} (n* = {n!r}); tr(A~* M~(w*)) = {trw!r}")
    log(f"C5 diagnostic (not a spec item): best mixture of the {m} evaluated policies with sum theta_i n_i <= B - 1: log det M - 197.197 = {pv-LD_JC!r} "
        f"(Clarabel {pst}; n of the mixture {nmix!r}; largest weights {supp}); D_best minus this = {D-pv!r} nat")
    np.save(os.path.join(RES, 'A_star.npy'), X); np.save(os.path.join(RES, 'A_star_unwhitened.npy'), unwhiten(X)); np.save(os.path.join(RES, 'w_star.npy'), w)
    res = dict(status=S['status'], iterations=m, best_iteration=b, D_best=D, D_best_minus_LDJC=D-LD_JC, gap=float(H['gap'][-1]),
               m_global=float(H['m_global'][-1]), norm_A_minus_I=nrm, eig_A=[float(v) for v in ev], lam_star=lam, lam_star_repr=repr(lam),
               lam_ratio_2c=lam/LAM_2C, n_star=n, n_minus_Bm1=n-BM1, n_within_5=bool(okn), max_cut_violation=vmax, m_global_min_step=mono,
               runtime_s=ttot, oracle_s=tor, master_s=tma, accepted=int(acc.sum()), rejected=int((1-acc).sum()), solver_counts=cnt,
               fwd_rel_max=float(Ca['fwd_rel'].max()), diag_logdet_Mw_star=ldw, diag_trace=trw, diag_mixture=dict(value=pv, n=nmix, status=pst,
               support=supp), files=dict(A_star='results/A_star.npy', A_star_unwhitened='results/A_star_unwhitened.npy', w_star='results/w_star.npy'))
    json.dump(res, open(os.path.join(RES, 'final.json'), 'w'), indent=1)

def stage_table(log):
    f = json.load(open(os.path.join(RES, 'final.json'))); r7 = ld.R7[ld.UHAT]
    rows = [('真实已评估最优（P5514，evaluator_fast B，n = 24；Δm_D 相对评价器的 log det J(u_unif)）', r7['dmD'], '评估值',
             'round2a/results/recheck7.json'),
            ('本轮 min D − 197.197（二维，A 优化）', f['D_best_minus_LDJC'], '估算', 'logs/final.log'),
            ('S 路线 U(619.129) − 197.197（2c）', S_ROUTE, '估算', 'round2c/results/part_c2.json'),
            ('固定 A = A_1（二维，2c）', DM_2C, '估算', 'round2c/results/outer_2d_A1.json')]
    log("D | 量 | 值（nat） | 性质 | 来源 |")
    for lab, v, kind, src in rows: log(f"D | {lab} | {v!r} | {kind} | {src} |")
    log(f"D P5514 on the - 197.197 scale: log det J(P5514) - 197.197 = {r7['logdet']!r} - 197.197 = {r7['logdet']-LD_JC!r} "
        f"(the +0.701 is relative to the evaluator's log det J(u_unif) = 197.20097668224076; offset 0.0044, round 2c DECISION A-5)")
    d = f['D_best_minus_LDJC']
    log(f"D differences: this round - S route = {d-S_ROUTE!r}; this round - fixed A = {d-DM_2C!r}; this round - (+0.701) = {d-r7['dmD']!r}; "
        f"S route - this round = {S_ROUTE-d!r} (M3: >= 0.3)")
    json.dump(dict(rows=[dict(label=a, value=b, kind=c, source=s) for a, b, c, s in rows], diff_S=d-S_ROUTE, diff_fixedA=d-DM_2C,
                   diff_real=d-r7['dmD'], P5514_on_LDJC_scale=r7['logdet']-LD_JC), open(os.path.join(RES, 'part_d.json'), 'w'), indent=1, ensure_ascii=False)

def stage_plot(log):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#898781', '#fcfcfb', '#e4e3df'
    BLUE, ORANGE = '#2a78d6', '#eb6834'
    z = np.load(CUTS); D = z['D']-LD_JC; acc = z['h_accepted']; mg = z['h_m_global']-LD_JC; Db = z['h_D_best']-LD_JC
    gap = z['h_gap']; r = z['h_r']; lam = z['lam']/LAM_2C; j = np.arange(D.size)
    fig, axs = plt.subplots(4, 1, figsize=(11, 11), sharex=True, constrained_layout=True, gridspec_kw=dict(height_ratios=[3, 1.4, 1, 1]))
    fig.patch.set_facecolor(SURF)
    def style(ax):
        ax.set_facecolor(SURF); ax.tick_params(labelsize=8, colors=INK2)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'): ax.spines[sp].set_color(MUTED)
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax = axs[0]; lo_ = min(float(mg[-1]), float(Db.min()))-0.6; hi_ = max(DM_2C, S_ROUTE, float(Db[0]))+0.8
    ax.set_ylim(lo_, hi_)
    for yv, lab in ((S_ROUTE, f'S route (2c) {S_ROUTE:.3f}'), (DM_2C, f'fixed A = A₁ (2c) {DM_2C:.3f}')):
        ax.axhline(yv, color=MUTED, lw=1.0, ls=(0, (4, 3))); ax.annotate(lab, (j[-1], yv), xytext=(4, 3), textcoords='offset points', fontsize=8, color=INK2, ha='right')
    ax.plot(j, Db, color=BLUE, lw=2, label='D_best − 197.197', solid_capstyle='round')
    ax.plot(j, mg, color=ORANGE, lw=2, label='m_global − 197.197 (cut model, no trust region)', solid_capstyle='round')
    ins = (D >= lo_) & (D <= hi_)
    ax.scatter(j[ins & (acc > 0)], D[ins & (acc > 0)], s=22, color=BLUE, edgecolor=SURF, linewidth=1.2, zorder=3, label='D_j, accepted step / start')
    ax.scatter(j[ins & (acc == 0)], D[ins & (acc == 0)], s=22, facecolor=SURF, edgecolor=BLUE, linewidth=1.2, zorder=3, label='D_j, rejected step')
    up = D > hi_
    if up.any(): ax.scatter(j[up], np.full(up.sum(), hi_), marker='^', s=22, color=MUTED, clip_on=False, zorder=3, label=f'D_j above {hi_:.1f} (off scale)')
    lw_ = D < lo_
    if lw_.any(): ax.scatter(j[lw_], np.full(lw_.sum(), lo_), marker='v', s=22, color=MUTED, clip_on=False, zorder=3)
    ax.annotate(f'{Db[-1]:.3f}', (j[-1], Db[-1]), xytext=(6, -2), textcoords='offset points', fontsize=8, color=INK2)
    ax.annotate(f'{mg[-1]:.3f}', (j[-1], mg[-1]), xytext=(6, -8), textcoords='offset points', fontsize=8, color=INK2)
    ax.set_ylabel('nat (value − 197.197)', fontsize=9, color=INK2); style(ax); ax.legend(fontsize=8, frameon=False, loc='upper right', ncol=2)
    ax.set_title('Round 2d: cutting-plane outer loop over (Ã, λ) — 估算 on the discrete 2-D model', fontsize=10, color=INK, loc='left')
    ax = axs[1]; ax.semilogy(j, np.maximum(gap, 1e-6), color=BLUE, lw=2); ax.axhline(GAP_STOP, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.annotate(f'stop at {GAP_STOP}', (0, GAP_STOP), xytext=(2, 3), textcoords='offset points', fontsize=8, color=INK2)
    ax.set_ylabel('gap D_best − m_global', fontsize=9, color=INK2); style(ax)
    ax = axs[2]; ax.semilogy(j, r, color=BLUE, lw=2, drawstyle='steps-post'); ax.set_ylabel('trust radius r', fontsize=9, color=INK2); style(ax)
    ax = axs[3]; ax.plot(j, lam, color=BLUE, lw=0, marker='o', ms=3.5); ax.set_yscale('symlog', linthresh=0.1)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3))); ax.set_ylabel('λ_j / λ*_2c', fontsize=9, color=INK2); style(ax)
    ax.set_xlabel('iteration j (oracle call)', fontsize=9, color=INK2)
    fig.savefig(os.path.join(HERE, 'fig_convergence.png'), dpi=130, facecolor=SURF); plt.close(fig)
    log(f"PLOT fig_convergence.png from results/cuts.npz ({D.size} iterations); panel 1 y range [{lo_:.3f}, {hi_:.3f}], {int(up.sum())} D_j above it")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['outer', 'final', 'table', 'plot'])
    ap.add_argument('--threads', type=int, default=None)
    args = ap.parse_args()
    if args.threads: set_num_threads(args.threads)
    logname = {'outer': 'outer.log', 'final': 'final.log', 'table': 'part_d.log', 'plot': 'plot.log'}[args.stage]
    sys.stdout = dr.Tee(os.path.join(LOGS, logname), 'a')
    def log(m): print(m, flush=True)
    of = open(os.path.join(LOGS, 'oracle.log'), 'a')
    def olog(m): of.write(m+'\n'); of.flush()
    T0 = time.time()
    hdr = (f"\n=== logdet_outer.py {args.stage} started {time.strftime('%Y-%m-%d %H:%M:%S')} (round 2d; 估算 on the discrete model); "
           f"numba threads {get_num_threads()}; load {g.loadavg()}")
    log(hdr)
    if args.stage == 'outer': olog(hdr); stage_outer(log, olog)
    elif args.stage == 'final': stage_final(log)
    elif args.stage == 'table': stage_table(log)
    else: stage_plot(log)
    log(f"=== logdet_outer.py {args.stage} finished {time.strftime('%Y-%m-%d %H:%M:%S')}; wall {time.time()-T0:.1f} s")
