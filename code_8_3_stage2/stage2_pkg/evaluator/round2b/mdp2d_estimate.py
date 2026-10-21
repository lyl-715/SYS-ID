"""Round 2b, Part A: 2-D closed-loop DP with the acceleration constraint (spec reports_CN/plan_round2b_CN.md v3.2, frozen, §1-§3, §5).

Everything is a discrete-model estimate (离散 MDP 对偶估算值) or a value of a policy on the discrete model, not a bound.

Model. State (delta, v_prev) on delta/sigma = -60 + i/30 (i = 0..3600, G = 3601, h = sigma/30) x v_prev = l h (|l| <= L = 137, 275
columns c = l + 137); 990275 states. Action: the new velocity v = l' h with l' - l_prev in {-1, 0, +1} (|l'| <= L). Transition:
no send (prob P(delta)) -> (delta - v, v); send (prob q(delta)) -> (e - v, v), e the discretised Gaussian masked by |e - delta| >= Delta
and renormalised per row; ends map to the end nodes. Reward r_lambda = c - lambda q, c = mc83.cfun. Backward over k = K..2 with
V_(K+1) = 0:  V_k(i, l_prev) = r_i + max_{|l' - l_prev| <= A} Q_k(i, l'),  Q_k(i, l') = P_i V_(k+1)(i - l', l') + q_i E[V_(k+1)(e - l', l') | send].
Q_k(., l') is the round-2a transition applied to the column V_(k+1)(., l'): round2a/mdp_estimate._q_action, imported unchanged and
called once per column (prefix/suffix sums per v column). The window max over l' uses the van Herk / Gil-Werman block algorithm for any
half-width A (A = 1: acceleration constraint; A = 2L: no constraint, for check 3.1). Initial state delta_2 = e_1 - v_1, v_1 any grid
velocity: V~ = max_{l_1} sum_j phat_j V_2(j - l_1, l_1) (round2a/mdp_estimate._init_value per column, same order and tie rule).
Ties of the policy (A = 1) are broken as in round 2a: smallest resulting |v| first, then +v before -v.

Stages (python3 mdp2d_estimate.py <stage>; every stage appends to logs/mdp2d_estimate.log):
  checks           A0 constants, A1 grids, 3.1 same-model regression (must pass), 3.3 forward-backward consistency (must pass)
  outer [--eps E]  A4: golden section on [0, 20] to bracket width E (default 1e-2); cache logs/gold_2d.jsonl (resumable)
  policy --lam X   A5/A6 at lambda = X: actions of all steps 2-bit packed into the disk memmap policy2d.u8 (k = 2..K, row k - 2,
                   state index c*G + i, code = Delta j + 1), forward S, E[N], budget, T3 statistics, plots fig_policy2d.png; then pack
  plot             figures again from the stored memmap (forward S, E[N] logged for comparison)
  pack / unpack    lossless git-sized copy results/policy2d_xordelta.npz (row XOR previous row, zlib) <-> policy2d.u8, sha256-checked
Usage (from this directory): nohup python3 mdp2d_estimate.py checks > logs/mdp2d_checks.stdout 2>&1 &"""
import os, sys, json, time, argparse, hashlib
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads
import grid30 as g
from grid30 import mdp, dr, G, M, L, K, NL, LH, LAM_BUDGET, LOGS, RES, HERE

_q_action = mdp._q_action; _clip = mdp._clip; _tails = mdp._tails
NB = (NL*G+3)//4                                        # bytes per step of the packed policy (4 codes per byte)
MEMMAP = os.path.join(HERE, 'policy2d.u8'); MEMMETA = os.path.join(RES, 'policy2d_meta.json')
PACKED = os.path.join(RES, 'policy2d_xordelta.npz')    # committed lossless copy of the memmap (row XOR previous row, zlib per row)
PLOT_KS = (500, 1250, 2500, 4000)


# --- kernels ------------------------------------------------------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _q_all(V, P, q, phat, Ninv, M, Lh, Q):
    """Q[c] = round-2a _q_action on the column V[c] with shift l = c - Lh (one call per v column)."""
    nl, n = V.shape
    for c in prange(nl):
        A = np.empty(n); Cl = np.empty(n+1); Sr = np.empty(n+1)
        _q_action(V[c], P, q, phat, Ninv, M, c-Lh, A, Cl, Sr, Q[c])

@njit(parallel=True, cache=True)
def _wmax(Q, A, r, Vout, Gp, Hs):
    """Vout[p, i] = max_{|l - p| <= A, 0 <= l < nl} Q[l, i] + r_i (van Herk / Gil-Werman on the padded axis t = l + A, blocks of
    w = 2A + 1; window of p = padded [p, p + 2A] = max(Hs[p], Gp[p + 2A])). Parallel over chunks of i."""
    nl, n = Q.shape; w = 2*A+1; npad = nl+2*A; nch = 64; csz = (n+nch-1)//nch
    for ch in prange(nch):
        i0 = ch*csz; i1 = min(n, i0+csz)
        for t in range(npad):
            l = t-A
            for i in range(i0, i1):
                v = Q[l, i] if (l >= 0 and l < nl) else -np.inf
                if t % w == 0 or v > Gp[t-1, i]: Gp[t, i] = v
                else: Gp[t, i] = Gp[t-1, i]
        for t in range(npad-1, -1, -1):
            l = t-A
            for i in range(i0, i1):
                v = Q[l, i] if (l >= 0 and l < nl) else -np.inf
                if t % w == w-1 or t == npad-1 or v > Hs[t+1, i]: Hs[t, i] = v
                else: Hs[t, i] = Hs[t+1, i]
        for p in range(nl):
            for i in range(i0, i1):
                a = Hs[p, i]; b = Gp[p+2*A, i]
                Vout[p, i] = (a if a > b else b)+r[i]

@njit(parallel=True, cache=True)
def _codes(Q, Vout, r, Lh, codes):
    """A = 1 policy: code = Delta j + 1 of the first maximiser in the order smallest |l'| first, then +l' before -l' (round-2a rule);
    returns the number of states where Q at the chosen action + r differs from Vout (must be 0)."""
    nl, n = Q.shape; bad = 0
    for p in prange(nl):
        lp = p-Lh; cand = np.empty(3, np.int64); nc = 0
        if lp > 0: order = (lp-1, lp, lp+1)
        elif lp < 0: order = (lp+1, lp, lp-1)
        else: order = (0, 1, -1)
        for u in range(3):
            if order[u] >= -Lh and order[u] <= Lh: cand[nc] = order[u]; nc += 1
        for i in range(n):
            best = -np.inf; bc = 1
            for u in range(nc):
                v = Q[cand[u]+Lh, i]
                if v > best: best = v; bc = cand[u]-lp+1
            codes[p, i] = bc
            if best+r[i] != Vout[p, i]: bad += 1
    return bad

@njit(parallel=True, cache=True)
def _apply_codes(Q, codes, r, Vout):
    """fixed policy: Vout[p, i] = Q[p + code - 1, i] + r_i."""
    nl, n = Q.shape
    for p in prange(nl):
        for i in range(n): Vout[p, i] = Q[p+codes[p, i]-1, i]+r[i]

@njit(parallel=True, cache=True)
def _pack(cf, outb):
    for b in prange(outb.size):
        v = 0
        for u in range(4):
            s = 4*b+u
            if s < cf.size: v |= (cf[s] & 3) << (2*u)
        outb[b] = v

@njit(parallel=True, cache=True)
def _unpack(inb, cf):
    for s in prange(cf.size): cf[s] = (inb[s >> 2] >> (2*(s & 3))) & 3

@njit(parallel=True, cache=True)
def _fwd(mu, codes, P, q, phat, Ninv, M, Lh, out):
    """mu_(k+1) from mu_k under the A = 1 policy codes (destination column c gathers the sources c - d with code d + 1)."""
    nl, n = mu.shape
    for c in prange(nl):
        l = c-Lh
        for i in range(n): out[c, i] = 0.0
        A = np.zeros(n); Cl = np.empty(n+1); Sr = np.empty(n+1); hit = False
        for d in (-1, 0, 1):
            cp = c-d
            if cp < 0 or cp >= nl: continue
            for i in range(n):
                if codes[cp, i] == d+1:
                    m = mu[cp, i]
                    if m != 0.0:
                        out[c, _clip(i-l, n)] += m*P[i]; A[i] += m*q[i]*Ninv[i]; hit = True
        if hit:
            _tails(A, Cl, Sr)
            for j in range(n):
                s = (Cl[j-M+1] if j-M >= 0 else 0.0)+(Sr[j+M] if j+M <= n-1 else 0.0)
                out[c, _clip(j-l, n)] += phat[j]*s


# --- drivers ---------------------------------------------------------------------------------------------------------------------------
def init_value(V2, phat):
    """V~ = max over l_1 (round-2a ORDER, strict >) of sum_j phat_j V_2(clip(j - l_1), l_1), each sum by round-2a _init_value."""
    best = -np.inf; bl = 0
    for l in g.ORDER:
        s_, _ = mdp._init_value(V2[l+LH], phat, np.array([l], np.int64))
        if s_ > best: best = s_; bl = int(l)
    return float(best), bl

def dp2d(lam, A, model, store=None, keep_V2=False):
    """backward over k = K..2 with V_(K+1) = 0; A = half-width of the action window. store: writable memmap (K-1, NB) for the
    packed A = 1 policy. Returns V~, v_1 (index), number of max mismatches, V_2 (if keep_V2)."""
    t, c, q, P, phat, Ninv, N = model
    r = c-lam*q
    V = np.zeros((NL, G)); Vout = np.empty((NL, G)); Q = np.empty((NL, G))
    npad = NL+2*A; Gp = np.empty((npad, G)); Hs = np.empty((npad, G))
    codes = np.empty((NL, G), np.int8); pk = np.empty(NB, np.uint8); bad = 0
    for k in range(K, 1, -1):
        _q_all(V, P, q, phat, Ninv, M, LH, Q)
        _wmax(Q, A, r, Vout, Gp, Hs)
        if store is not None:
            bad += _codes(Q, Vout, r, LH, codes); _pack(codes.reshape(-1), pk); store[k-2] = pk
        V, Vout = Vout, V
    Vt, l1 = init_value(V, phat)
    return Vt, l1, bad, (V.copy() if keep_V2 else None)

def backward_fixed2d(codes_at, l1, rew, model):
    """fixed 2-D policy (codes_at(k) for k = 2..K-1; step K has Q = 0): sum_{k>=2} E[rew] read at the initial state (l_1)."""
    t, c, q, P, phat, Ninv, N = model
    V = np.zeros((NL, G)); Vout = np.empty((NL, G)); Q = np.empty((NL, G)); one = np.ones((NL, G), np.int8)
    for k in range(K, 1, -1):
        _q_all(V, P, q, phat, Ninv, M, LH, Q)
        _apply_codes(Q, codes_at(k) if k < K else one, rew, Vout)
        V, Vout = Vout, V
    return float(mdp._init_value(V[l1+LH], phat, np.array([l1], np.int64))[0])

def forward2d(codes_at, l1, model, keep_ks=()):
    """forward distribution mu_k(delta, v_prev), k = 2..K; S = 1 + sum_k E[c], E[N] = 1 + sum_k E[q]; T3 statistics per k."""
    t, c, q, P, phat, Ninv, N = model
    mu = np.zeros((NL, G)); mu[l1+LH] = mdp.initial_mu(phat, l1); out = np.empty((NL, G))
    bm = g.band_mask(); lowrows = np.abs(np.arange(NL)-LH) <= g.LOW
    S = 1.0; EN = 1.0; band = np.zeros(K+1); low = np.zeros(K+1); mass = np.zeros(K+1); kept = {}
    for k in range(2, K+1):
        S += float((mu@c).sum()); EN += float((mu@q).sum())
        band[k] = float(mu[:, bm].sum()); low[k] = float(mu[lowrows].sum()); mass[k] = float(mu.sum())
        if k in keep_ks: kept[k] = mu.copy()
        if k < K:
            _fwd(mu, codes_at(k), P, q, phat, Ninv, M, LH, out); mu, out = out, mu
    return S, EN, float(mu.sum()), band, low, mass, kept

def open_loop_codes(lfull):
    """spec §3.3 policy in 2-D form: at step k, code(i, l_prev) = clip(l_k - l_prev, -1, 1) + 1 for every state."""
    lp = np.arange(NL)-LH
    def at(k):
        d = np.clip(lfull[k-1]-lp, -1, 1).astype(np.int8)+1                      # lfull[k-1] = l_k (v_k), k = 1..K-1
        return np.ascontiguousarray(np.broadcast_to(d[:, None], (NL, G)))
    return at


def constants():
    vmax = dr.dt*float(np.sqrt(np.sum((dr.w*np.abs(dr.Gj))**2)))
    acc = dr.w**2*np.abs(dr.Gj); amax = dr.dt**2*float(np.sqrt(np.sum(acc**2)))
    return vmax, amax, acc


# --- stages ------------------------------------------------------------------------------------------------------------------------------
def stage_checks(model, log):
    vmax, amax, acc = constants(); s = dr.s; h = s/g.N_SIG
    log(f"A0 v_max := dt sqrt(sum_m (w_m |G(j w_m)|)^2) = {vmax!r} ≈ {vmax/s:.4f} sigma ({vmax/s!r} sigma); evaluator_fast.v_max() = {ef_vmax()!r}")
    log(f"A0 w_m^2 |G(j w_m)| = ({', '.join(f'{v:.3f}' for v in acc)}) (full: {', '.join(repr(float(v)) for v in acc)}); sqrt(sum of squares) ≈ {np.sqrt(np.sum(acc**2)):.4f}")
    log(f"A0 a_max := dt^2 sqrt(sum_m (w_m^2 |G(j w_m)|)^2) = {amax!r} ≈ {amax:.5e} ≈ {amax/s:.6f} sigma ({amax/s!r} sigma); displayed values are ≈, "
        f"not conservative constants")
    log(f"A0 sigma = Delta/20 = {s!r}; dt = {dr.dt!r}; v_max/a_max ≈ {vmax/amax:.1f} steps to stop from v_max; K/316 ≈ {dr.K/316:.1f} steps per send")
    log(f"A1 h = sigma/{g.N_SIG} = {h!r} = {h/s!r} sigma; h > a_max: {h > amax} (h/a_max = {h/amax!r}); L = floor(v_max/h) = {int(np.floor(vmax/h))} "
        f"(used {L}; endpoint {L/g.N_SIG:.4f} sigma); offset grid G = {G}, delta/sigma in [-60, 60] (= [-Delta - 40 sigma, Delta + 40 sigma]), "
        f"M = Delta/h = {M}; velocity columns {NL}; states {NL*G}; actions Delta j in {{-1, 0, +1}}")
    t, c, q, P, phat, Ninv, N = model
    log(f"A2 model: min N_i {float(N.min())!r} (all > 0: {bool((N > 0).all())}); sum phat {float(phat.sum())!r}; c in [{float(c.min())!r}, {float(c.max())!r}]; "
        f"q in [{float(q.min())!r}, {float(q.max())!r}]")
    xU, _ = dr.input_arrays('U'); d2 = np.abs(np.diff(xU, 2))
    log(f"A0 check item: uniform input max |x_(k+2) - 2 x_(k+1) + x_k| = {float(d2.max())!r} = {float(d2.max())/amax!r} a_max (<= a_max: {bool(d2.max() <= amax)})")
    # --- 3.1 -------------------------------------------------------------------------------------------------------------------
    lam = g.LAM_STAR_2A; tb = time.time()
    Vt1, l11, pols = mdp.dual_value(lam, c, q, P, phat, Ninv)
    t1d = time.time()-tb
    # the 1-D V_2 array (round-2a _max_step loop, same as mdp.dual_value) for the state-wise comparison
    r = c-lam*q; V1 = r.copy(); Vo = np.empty(G); pol = np.zeros(G, np.int64)
    for k in range(K-1, 1, -1):
        mdp._max_step(V1, r, P, q, phat, Ninv, M, g.ORDER, Vo, pol); V1, Vo = Vo, V1
    tb = time.time(); Vt2, l12, bad, V2 = dp2d(lam, 2*L, model, keep_V2=True); t2d = time.time()-tb
    rel = abs(Vt2-Vt1)/abs(Vt1); dV = float(np.abs(V2-V1[None, :]).max()); ok31 = rel <= 1e-10
    log(f"3.1 same-model regression at the archived round-2a lambda* = {lam!r}, same grid (h = sigma/30, L = {L}), same kernel: 1-D V~ {Vt1!r} "
        f"(v_1 = {l11} h; {t1d:.1f} s) vs 2-D without the acceleration constraint (window half-width A = 2L = {2*L}) V~ {Vt2!r} (v_1 = {l12} h; {t2d:.1f} s)")
    log(f"3.1 relative difference of V~ {rel!r} (absolute {abs(Vt2-Vt1)!r}); <= 1e-10: {ok31}; bitwise equal: {Vt1 == Vt2}; "
        f"max over all states (i, l_prev) of |V_2^2D - V_2^1D| = {dV!r}")
    # --- 3.3 -------------------------------------------------------------------------------------------------------------------
    lfull, rerr, inc = g.uniform_open_loop(); l1 = int(lfull[0]); dl = np.abs(np.diff(lfull))
    log(f"3.3 open-loop policy: uniform input's increments rounded to multiples of h: |l| max {int(np.abs(lfull).max())} (<= L: {bool(np.abs(lfull).max() <= L)}); "
        f"max rounding error {rerr!r} h; max |l_(k+1) - l_k| = {int(dl.max())} (<= 1: {bool(dl.max() <= 1)}); max true |inc_(k+1) - inc_k| = "
        f"{float(np.abs(np.diff(inc)).max())!r} h; v_1 = {l1} h")
    at = open_loop_codes(lfull); tb = time.time()
    Sf, ENf, massf, band, low, mass, _ = forward2d(at, l1, model)
    Sb = 1.0+backward_fixed2d(at, l1, c, model); ENb = 1.0+backward_fixed2d(at, l1, q, model)
    rS, rN = abs(Sf-Sb)/abs(Sb), abs(ENf-ENb)/abs(ENb); ok33 = rS <= 1e-10 and rN <= 1e-10
    log(f"3.3 2-D forward S {Sf!r} E[N] {ENf!r} (final mass {massf!r}); 2-D fixed-policy backward S {Sb!r} E[N] {ENb!r}; {time.time()-tb:.1f} s")
    log(f"3.3 relative difference S {rS!r}, E[N] {rN!r}; <= 1e-10: {ok33}")
    lseq = np.zeros(K+1, np.int64); lseq[2:K] = lfull[1:K-1]
    S1, EN1, _ = mdp.forward(lambda k: np.full(G, lseq[k], np.int64), l1, c, q, P, phat, Ninv)
    log(f"3.3 diagnostic (no pass line): 1-D forward of the same open-loop policy on the same grid S {S1!r} E[N] {EN1!r}; 2-D - 1-D: S {Sf-S1!r}, E[N] {ENf-EN1!r}")
    out = dict(constants=dict(v_max=vmax, v_max_sigma=vmax/s, a_max=amax, a_max_sigma=amax/s, h_sigma=1/g.N_SIG, L=L),
               check_3_1=dict(lam=lam, V1=Vt1, V2=Vt2, rel=rel, abs=abs(Vt2-Vt1), max_state_abs=dV, passed=bool(ok31), t1d=t1d, t2d=t2d),
               check_3_3=dict(S_forward=Sf, S_backward=Sb, EN_forward=ENf, EN_backward=ENb, rel_S=rS, rel_EN=rN, passed=bool(ok33),
                              S_1d=S1, EN_1d=EN1))
    json.dump(out, open(os.path.join(RES, 'mdp2d_checks.json'), 'w'), indent=1)
    if not (ok31 and ok33): log("CHECKS FAILED: STOPPED (no 2-D estimate is reported)")
    return ok31 and ok33

def ef_vmax():
    import evaluator_fast as ef
    return ef.v_max()

def stage_outer(model, eps, log):
    cache = g.Cache(os.path.join(LOGS, 'gold_2d.jsonl'))
    def Vlam(lam):
        r = cache.get(lam)
        if r is None:
            ts = time.time(); Vt, l1, _, _ = dp2d(lam, 1, model); dt_ = time.time()-ts
            r = dict(V=Vt, l1=l1, time_s=dt_, threads=get_num_threads(), load=g.loadavg()); cache.put(lam, r)
            log(f"GOLD-2D lambda {lam!r} V~ {Vt!r} V~ + {LAM_BUDGET} lambda {Vt+LAM_BUDGET*lam!r} (v_1 = {l1} h; {dt_:.1f} s; numba threads {r['threads']}; "
                f"load {r['load']})")
        else:
            log(f"GOLD-2D lambda {lam!r} V~ {r['V']!r} V~ + {LAM_BUDGET} lambda {r['V']+LAM_BUDGET*lam!r} (v_1 = {r['l1']} h; cached from logs/gold_2d.jsonl, "
                f"computed in {r['time_s']:.1f} s)")
        return r['V'], r['l1']
    tg = time.time()
    evals, lam_s, Fmin, lo, hi = g.golden(Vlam, eps, log)
    Sd = 1+Fmin; tsum = sum(cache.get(k)['time_s'] for k in evals)
    log(f"GOLD-2D final bracket [{lo!r}, {hi!r}] (width {hi-lo!r}); eps_lambda {eps!r}; {len(evals)} evaluations; wall of this call {time.time()-tg:.1f} s; "
        f"sum of per-lambda DP times {tsum:.1f} s ({tsum/3600:.3f} h)")
    log(f"A4-2D 离散 MDP 对偶估算值 (acceleration constraint) S~_acc = 1 + min_lambda {{V~ + {LAM_BUDGET} lambda}} = {Sd!r} at lambda* = {lam_s!r} "
        f"(V~ = {evals[lam_s][0]!r}; v_1 = {evals[lam_s][1]} h); eps_lambda = {eps!r}; an estimate on the discrete model, not a bound")
    res = dict(model='2-D, acceleration constraint (Delta j in {-1,0,1}), h = sigma/30, L = 137', eps_lambda=eps, lam_star=lam_s, lam_star_repr=repr(lam_s),
               V_at_lam_star=evals[lam_s][0], S_acc=Sd, n_evals=len(evals), bracket=[lo, hi], v1=evals[lam_s][1], dp_time_sum_s=tsum,
               evals={repr(k): v[0] for k, v in sorted(evals.items())})
    json.dump(res, open(os.path.join(RES, f'mdp2d_outer_eps{eps:.6g}.json'), 'w'), indent=1)
    return res

def stage_plot(model, log):
    """figures only, from the stored memmap: forward again (S, E[N] logged for comparison with the policy stage) and plot."""
    meta = json.load(open(MEMMETA)); lam = meta['lam']; l1 = meta['l1']
    mm = np.memmap(MEMMAP, dtype=np.uint8, mode='r', shape=(K-1, NB)); cf = np.empty(NL*G, np.int8)
    def at(k):
        _unpack(np.asarray(mm[k-2]), cf); return cf.reshape(NL, G)
    Sp, ENp, _, _, _, _, kept = forward2d(at, l1, model, keep_ks=PLOT_KS)
    log(f"PLOT from the stored memmap (lambda {meta['lam_repr']}, sha256 {meta['sha256']}): forward S {Sp!r}, E[N] {ENp!r}; figures fig_policy2d.png, fig_occupancy2d.png")
    plot_policy(at, kept, lam, Sp, ENp)

def stage_policy(model, lam, log):
    t, c, q, P, phat, Ninv, N = model
    tb = time.time()
    mm = np.memmap(MEMMAP+'.tmp', dtype=np.uint8, mode='w+', shape=(K-1, NB))
    Vt, l1, bad, _ = dp2d(lam, 1, model, store=mm)
    mm.flush(); del mm; os.replace(MEMMAP+'.tmp', MEMMAP); tdp = time.time()-tb
    hsh = hashlib.sha256()
    with open(MEMMAP, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 24), b''): hsh.update(blk)
    size = os.path.getsize(MEMMAP)
    log(f"A5 policy at lambda = {lam!r}: V~ {Vt!r} (v_1 = {l1} h); backward with storage {tdp:.1f} s; max mismatches (chosen action vs window max) {bad}")
    log(f"A5 memmap {os.path.basename(MEMMAP)}: shape ({K-1}, {NB}) uint8 = {size} bytes = {size/2**30:.4f} GiB; row k - 2 for k = 2..{K}; 4 codes per byte "
        f"(code = Delta j + 1, bits 2u..2u+1 hold state 4b + u, state = c*G + i, c = l_prev + {LH}); sha256 {hsh.hexdigest()}")
    json.dump(dict(lam=lam, lam_repr=repr(lam), file=os.path.basename(MEMMAP), shape=[K-1, NB], dtype='uint8', bytes=size, sha256=hsh.hexdigest(),
                   layout='row k-2 (k = 2..K); state s = c*G + i (c = l_prev + 137, i = delta index); byte s>>2, bits 2(s&3); code = Delta j + 1',
                   regenerate=f'python3 mdp2d_estimate.py policy --lam {lam!r}', V=Vt, l1=l1), open(MEMMETA, 'w'), indent=1)
    mm = np.memmap(MEMMAP, dtype=np.uint8, mode='r', shape=(K-1, NB)); cf = np.empty(NL*G, np.int8)
    def at(k):
        _unpack(np.asarray(mm[k-2]), cf); return cf.reshape(NL, G)
    tf = time.time()
    Sp, ENp, massp, band, low, mass, kept = forward2d(at, l1, model, keep_ks=PLOT_KS)
    log(f"A5 forward of the stored policy (discrete model): S {Sp!r}, E[N] {ENp!r} (final mass {massp!r}); {time.time()-tf:.1f} s")
    log(f"A5 E[N] <= B = {g.B_FULL!r}: {ENp <= g.B_FULL} (E[N] - B = {ENp-g.B_FULL!r}); reported, not a failure; lambda not adjusted")
    Vchk = (Sp-1)-lam*(ENp-1)
    log(f"A5 diagnostic (not a spec check): V~ {Vt!r} vs (S - 1) - lambda (E[N] - 1) = {Vchk!r}; relative {abs(Vchk-Vt)/abs(Vt)!r}")
    ks = np.arange(2, K+1); T3b, T3l = float(band[ks].mean()), float(low[ks].mean())
    log(f"T3-2D time average over k = 2..{K} (K - 1 = {K-1} steps) of Pr{{18 <= |delta|/sigma <= 19}} = {T3b!r} (nodes {g.BAND[0]} <= |i - {g.I0}| <= {g.BAND[1]})")
    log(f"T3-2D time average over k = 2..{K} of Pr{{|v_prev| <= 0.5 sigma}} = {T3l!r} (|l_prev| <= {g.LOW}; state component v_(k-1), k = 2..K)")
    log(f"T3-2D mass over k: min {float(mass[ks].min())!r} max {float(mass[ks].max())!r}")
    for kk in PLOT_KS:
        mu = kept[kk]; mv = mu.sum(1); vgrid = (np.arange(NL)-LH)/g.N_SIG; cd = at(kk)
        fr = [float(mu[cd == u].sum()) for u in range(3)]
        log(f"T3-2D at k = {kk}: Pr{{18 <= |delta|/sigma <= 19}} = {float(band[kk])!r}; Pr{{|v_prev| <= 0.5 sigma}} = {float(low[kk])!r}; "
            f"E|v_prev|/sigma = {float(mv@np.abs(vgrid))!r}; Pr{{v_prev = 0}} = {float(mv[LH])!r}; mu-weighted share of Delta j = -1/0/+1: "
            f"{fr[0]!r} / {fr[1]!r} / {fr[2]!r}")
    np.savez_compressed(os.path.join(RES, 'mdp2d_forward_stats.npz'), band=band, low=low, mass=mass, lam=lam)
    plot_policy(at, kept, lam, Sp, ENp)
    res = dict(lam=lam, lam_repr=repr(lam), V=Vt, v1=l1, S=Sp, EN=ENp, EN_le_B=bool(ENp <= g.B_FULL), B=g.B_FULL,
               T3=dict(band_18_19=T3b, low_speed=T3l), memmap=dict(file=os.path.basename(MEMMAP), bytes=size, sha256=hsh.hexdigest()),
               mismatches=int(bad), backward_store_s=tdp)
    json.dump(res, open(os.path.join(RES, 'mdp2d_policy.json'), 'w'), indent=1)
    stage_pack(log)
    return res

def sha256_file(path):
    hsh = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 24), b''): hsh.update(blk)
    return hsh.hexdigest()

def stage_pack(log):
    """lossless copy of policy2d.u8 small enough for git: row 0 as is, row r XOR row r-1 for r >= 1, each zlib level 6;
    decoded back into a temporary file and compared with the memmap's sha256 before it is accepted."""
    import zlib
    meta = json.load(open(MEMMETA)); mm = np.memmap(MEMMAP, dtype=np.uint8, mode='r', shape=(K-1, NB))
    chunks = []; prev = np.zeros(NB, np.uint8)
    for r_ in range(K-1):
        row = np.asarray(mm[r_]); chunks.append(zlib.compress(np.bitwise_xor(row, prev).tobytes(), 6)); prev = row
    lens = np.array([len(c_) for c_ in chunks], np.int64); blob = np.frombuffer(b''.join(chunks), np.uint8)
    np.savez(PACKED, lengths=lens, blob=blob, shape=np.array([K-1, NB]), sha256=np.array(meta['sha256']), lam=np.array(meta['lam']))
    unpack_policy(PACKED, MEMMAP+'.check.tmp'); h = sha256_file(MEMMAP+'.check.tmp'); os.remove(MEMMAP+'.check.tmp')
    ok = h == meta['sha256']
    log(f"A5 packed copy {os.path.relpath(PACKED, HERE)}: {os.path.getsize(PACKED)} bytes ({os.path.getsize(PACKED)/2**20:.2f} MiB) for lambda {meta['lam_repr']}; "
        f"decoded sha256 {h} equals the memmap's: {ok} (rebuild: python3 mdp2d_estimate.py unpack)")
    meta.update(packed=os.path.relpath(PACKED, HERE), packed_bytes=os.path.getsize(PACKED), packed_roundtrip_sha256_ok=bool(ok))
    json.dump(meta, open(MEMMETA, 'w'), indent=1)
    if not ok: raise RuntimeError('packed policy does not decode to the memmap')

def unpack_policy(src, dst):
    import zlib
    z = np.load(src); lens = z['lengths']; blob = z['blob']; nr, nb = (int(v) for v in z['shape'])
    mm = np.memmap(dst, dtype=np.uint8, mode='w+', shape=(nr, nb)); prev = np.zeros(nb, np.uint8); off = 0
    for r_ in range(nr):
        d = np.frombuffer(zlib.decompress(blob[off:off+lens[r_]].tobytes()), np.uint8); off += int(lens[r_])
        prev = np.bitwise_xor(d, prev); mm[r_] = prev
    mm.flush(); del mm
    return str(z['sha256'])

def plot_policy(at, kept, lam, Sp, ENp):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    ink, ink2 = '#0b0b0b', '#52514e'
    cmap = ListedColormap(['#3987e5', '#e4e3df', '#e34948']); norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
    dgrid = -g.HALF+np.arange(G)/g.N_SIG; vgrid = (np.arange(NL)-LH)/g.N_SIG
    md = sum(kept[kk].sum(0) for kk in PLOT_KS); sup = dgrid[md > 1e-6*md.max()]
    xr = max(24.0, float(np.ceil(np.abs(sup).max()))+1.0)          # shown range: [-24, 24] or the forward mass support + 1 sigma
    xs = (dgrid >= -xr) & (dgrid <= xr)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True, constrained_layout=True)
    for ax, kk in zip(axs.ravel(), PLOT_KS):
        cd = at(kk)
        ax.imshow(cd[:, xs], origin='lower', aspect='auto', cmap=cmap, norm=norm, interpolation='nearest',
                  extent=[dgrid[xs][0]-0.5/g.N_SIG, dgrid[xs][-1]+0.5/g.N_SIG, vgrid[0]-0.5/g.N_SIG, vgrid[-1]+0.5/g.N_SIG])
        mu = kept[kk][:, xs]; mx = mu.max()
        if mx > 0:
            ax.contour(dgrid[xs], vgrid, mu, levels=[1e-3*mx, 1e-2*mx, 1e-1*mx], colors=[ink, ink, ink], linewidths=[0.7, 1.0, 1.4])
        for x0 in (-20, 20): ax.axvline(x0, color=ink2, lw=0.6, ls=':')
        ax.set_title(f'k = {kk}', fontsize=10, color=ink, loc='left')
        ax.tick_params(labelsize=8, colors=ink2)
    for ax in axs[1]: ax.set_xlabel('δ / σ  (offset r − x)', fontsize=9, color=ink2)
    for ax in axs[:, 0]: ax.set_ylabel('v_prev / σ', fontsize=9, color=ink2)
    fig.legend(handles=[Patch(color='#3987e5', label='Δj = −1  (v* = v_prev − h)'), Patch(color='#e4e3df', label='Δj = 0  (v* = v_prev)'),
                        Patch(color='#e34948', label='Δj = +1  (v* = v_prev + h)')], loc='outside lower center', ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f'2-D policy v*(δ, v_prev) = v_prev + Δj·h at λ = {lam:.6f} (discrete model; forward S = {Sp:.2f}, E[N] = {ENp:.2f})\n'
                 f'contours: forward distribution μ_k at 1e-3, 1e-2, 1e-1 of its max; dotted: δ = ±Δ', fontsize=10, color=ink, x=0.01, ha='left')
    fig.savefig(os.path.join(HERE, 'fig_policy2d.png'), dpi=130); plt.close(fig)
    # occupancy marginals
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    seq = ['#9ec5f4', '#5598e7', '#256abf', '#0d366b']
    for kk, col in zip(PLOT_KS, seq):
        mu = kept[kk]
        axs[0].plot(dgrid, mu.sum(0)*g.N_SIG, color=col, lw=1.5, label=f'k = {kk}')
        axs[1].plot(vgrid, mu.sum(1)*g.N_SIG, color=col, lw=1.5, label=f'k = {kk}')
    axs[0].set_xlim(-xr, xr); axs[0].set_xlabel('δ / σ', fontsize=9, color=ink2); axs[0].set_ylabel('density per σ', fontsize=9, color=ink2)
    axs[1].set_xlabel('v_prev / σ', fontsize=9, color=ink2)
    for ax in axs:
        ax.tick_params(labelsize=8, colors=ink2); ax.grid(alpha=0.25, lw=0.5)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    axs[0].axvspan(-19, -18, color='#e4e3df', zorder=0); axs[0].axvspan(18, 19, color='#e4e3df', zorder=0)
    axs[1].axvspan(-0.5, 0.5, color='#e4e3df', zorder=0)
    axs[1].legend(fontsize=8, frameon=False)
    axs[0].set_title('marginal of μ_k over δ (shaded: 18 ≤ |δ|/σ ≤ 19)', fontsize=9, color=ink, loc='left')
    axs[1].set_title('marginal of μ_k over v_prev (shaded: |v_prev| ≤ 0.5σ)', fontsize=9, color=ink, loc='left')
    fig.savefig(os.path.join(HERE, 'fig_occupancy2d.png'), dpi=130); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['checks', 'outer', 'policy', 'all', 'pack', 'unpack', 'plot'])
    ap.add_argument('--eps', type=float, default=1e-2); ap.add_argument('--lam', type=float, default=None); ap.add_argument('--threads', type=int, default=None)
    args = ap.parse_args()
    if args.threads: set_num_threads(args.threads)
    sys.stdout = dr.Tee(os.path.join(LOGS, 'mdp2d_estimate.log'), 'a')
    def log(m): print(m, flush=True)
    T0 = time.time()
    log(f"\n=== mdp2d_estimate.py {args.stage} started {time.strftime('%Y-%m-%d %H:%M:%S')} (round 2b Part A; 离散 MDP 对偶估算值, not a bound); "
        f"numba threads {get_num_threads()}; load average {g.loadavg()}")
    if args.stage == 'pack': stage_pack(log); sys.exit(0)
    if args.stage == 'unpack':
        ref = unpack_policy(PACKED, MEMMAP+'.tmp'); h = sha256_file(MEMMAP+'.tmp'); os.replace(MEMMAP+'.tmp', MEMMAP)
        log(f"unpack: {os.path.relpath(PACKED, HERE)} -> {os.path.basename(MEMMAP)}; sha256 {h}; equals the archived {ref}: {h == ref}"); sys.exit(0)
    model = mdp.model()
    if args.stage in ('checks', 'all'):
        ok = stage_checks(model, log)
        if not ok: sys.exit(1)
    if args.stage in ('outer', 'all'):
        res = stage_outer(model, args.eps, log)
        if args.stage == 'all': args.lam = res['lam_star']
    if args.stage in ('policy', 'all'):
        stage_policy(model, args.lam, log)
    if args.stage == 'plot': stage_plot(model, log)
    log(f"=== mdp2d_estimate.py {args.stage} finished {time.strftime('%Y-%m-%d %H:%M:%S')}; wall {time.time()-T0:.1f} s")
