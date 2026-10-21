"""Round 2b, A3.2 / A3.3 (1-D) / A4 (1-D) / A6 (1-D): the round-2a 1-D closed-loop model, no acceleration constraint, on the new grid
(h = sigma/30, L = 137; grid30.py). Spec reports_CN/plan_round2b_CN.md v3.2 (frozen), §2, §3.2, §3.4, §6 T3.

Everything is a discrete-model estimate (离散 MDP 对偶估算值), not a bound. The DP, forward and fixed-policy backward are the round-2a
functions of round2a/mdp_estimate.py, imported unchanged (grid30.py sets the grid).
  3.3 (1-D)  forward vs fixed-policy backward of the rounded open-loop uniform policy, S and E[N], relative 1e-10 (must pass)
  3.2        cross-grid diagnostic, no pass line, 1 % attention line: (a) at the archived round-2a lambda* = 1.8335414994039494
             (same lambda, grid changed): 1 + V~_new(lambda*) + 315.892 lambda* vs 1210.956 = 1 + V~_old(lambda*) + 315.892 lambda*;
             (b) the outer-minimised new-grid value (A4) vs 1210.956
  A4 (1-D)   golden section on [0, 20] to bracket width eps = 1e-2; S~_dual = 1 + min{V~ + 315.892 lambda}
  A6 (1-D)   policy at lambda*: forward S, E[N]; T3 time averages over k = 2..K of Pr{18 <= |delta|/sigma <= 19} and
             Pr{|v_(k-1)| <= 0.5 sigma} (v_1 = initial velocity; v_(k-1) = pol_(k-1)(delta_(k-1)) for k >= 3)
Writes logs/mdp1d_newgrid.log, logs/gold_1d.jsonl (cache), results/mdp1d_result.json, results/mdp1d_policy.npz.
Usage (from this directory): nohup python3 mdp1d_newgrid.py > logs/mdp1d.stdout 2>&1 &"""
import os, sys, json, time
import numpy as np
import grid30 as g
from grid30 import mdp, dr, G, M, L, K, LAM_BUDGET, LOGS, RES

EPS = 1e-2


def forward_t3(pols, l1, c, q, P, phat, Ninv):
    """mdp.forward with the T3 statistics recorded per k (same transport, mdp._forward_step; same S, E[N] arithmetic)."""
    mu = mdp.initial_mu(phat, l1); S = 1.0; EN = 1.0; out = np.empty(G); used = np.zeros(2*L+1, np.bool_)
    bm = g.band_mask(); band = np.zeros(K+1); low = np.zeros(K+1); mass = np.zeros(K+1)
    vprev_low = None
    for k in range(2, K+1):
        S += float(mu@c); EN += float(mu@q)
        band[k] = float(mu[bm].sum()); mass[k] = float(mu.sum())
        low[k] = (1.0 if abs(l1) <= g.LOW else 0.0) if k == 2 else vprev_low
        if k < K:
            pk = pols[k].astype(np.int64)
            vprev_low = float(mu[np.abs(pk) <= g.LOW].sum())               # Pr{|v_k| <= 0.5 sigma}, the state component at k+1
            used[:] = False; used[np.unique(pk)+L] = True
            mdp._forward_step(mu, P, q, phat, Ninv, M, pk, used, out); mu, out = out, mu
    return S, EN, float(mu.sum()), band, low, mass


if __name__ == '__main__':
    sys.stdout = dr.Tee(os.path.join(LOGS, 'mdp1d_newgrid.log'), 'a')
    T0 = time.time()
    t, c, q, P, phat, Ninv, N = mdp.model()
    print(f"\n=== mdp1d_newgrid.py started {time.strftime('%Y-%m-%d %H:%M:%S')} (round 2b; 1-D model without acceleration constraint; "
          f"离散 MDP 对偶估算值, not a bound); grid G = {G} (delta/sigma in [-60, 60], h = sigma/{g.N_SIG}), M = {M}, actions l = -{L}..{L} "
          f"({L} h = {L/g.N_SIG:.4f} sigma), K = {K}, Lagrangian term {LAM_BUDGET} lambda; load average {g.loadavg()}")
    print(f"MODEL min N_i {float(N.min())!r} (all > 0: {bool((N > 0).all())}); t[0] {t[0]!r} t[-1] {t[-1]!r}; c in [{float(c.min())!r}, {float(c.max())!r}]")
    # --- 3.3 (1-D) -----------------------------------------------------------------------------------------------------------
    lfull, rerr, _ = g.uniform_open_loop()
    l1 = int(lfull[0]); lseq = np.zeros(K+1, np.int64); lseq[2:K] = lfull[1:K-1]
    print(f"3.3-1D open-loop policy: uniform input's increments rounded to multiples of h: |l| max {int(np.abs(lfull).max())} (<= L = {L}: "
          f"{bool(np.abs(lfull).max() <= L)}); max rounding error {rerr!r} h; v_1 = {l1} h")
    tb = time.time()
    Sf, ENf, mass = mdp.forward(lambda k: np.full(G, lseq[k], np.int64), l1, c, q, P, phat, Ninv)
    Sb = 1.0+mdp.backward_fixed(lseq, l1, c, P, q, phat, Ninv); ENb = 1.0+mdp.backward_fixed(lseq, l1, q, P, q, phat, Ninv)
    rS, rN = abs(Sf-Sb)/abs(Sb), abs(ENf-ENb)/abs(ENb); ok33 = rS <= 1e-10 and rN <= 1e-10
    print(f"3.3-1D forward S {Sf!r} E[N] {ENf!r} (final mass {mass!r}); backward S {Sb!r} E[N] {ENb!r}")
    print(f"3.3-1D relative difference S {rS!r}, E[N] {rN!r}; <= 1e-10: {ok33}; time {time.time()-tb:.1f} s")
    recU = json.load(open(os.path.join(g.EV, 'results', 'rec_B_U_n24_m8.json')))
    print(f"3.3-1D diagnostic (no pass line): forward vs evaluator uniform input (B, n = 24): S {Sf!r} vs {recU['S']!r} (relative "
          f"{(Sf-recU['S'])/recU['S']!r}); E[N] {ENf!r} vs {recU['EN']!r} (relative {(ENf-recU['EN'])/recU['EN']!r})")
    if not ok33:
        print("3.3-1D FAILED: STOPPED (no 1-D estimate is reported)"); sys.exit(1)
    # --- 3.2 (a): same lambda, new grid -----------------------------------------------------------------------------------------
    cache = g.Cache(os.path.join(LOGS, 'gold_1d.jsonl'))
    def Vlam(lam):
        r = cache.get(lam)
        if r is None:
            ts = time.time(); Vt, l1_, _ = mdp.dual_value(lam, c, q, P, phat, Ninv); dt_ = time.time()-ts
            r = dict(V=float(Vt), l1=int(l1_), time_s=dt_, load=g.loadavg()); cache.put(lam, r)
            print(f"GOLD-1D lambda {lam!r} V~ {r['V']!r} V~ + {LAM_BUDGET} lambda {r['V']+LAM_BUDGET*lam!r} (v_1 = {r['l1']} h; {dt_:.1f} s; load {r['load']})")
        else:
            print(f"GOLD-1D lambda {lam!r} V~ {r['V']!r} V~ + {LAM_BUDGET} lambda {r['V']+LAM_BUDGET*lam!r} (v_1 = {r['l1']} h; cached from "
                  f"logs/gold_1d.jsonl, computed in {r['time_s']:.1f} s)")
        return r['V'], r['l1']
    V2a, _ = Vlam(g.LAM_STAR_2A); S2a = 1+V2a+LAM_BUDGET*g.LAM_STAR_2A
    d32a = S2a-g.S_DUAL_2A
    print(f"3.2(a) same lambda* = {g.LAM_STAR_2A!r} (archived, round 2a): new grid 1 + V~ + {LAM_BUDGET} lambda* = {S2a!r} vs old grid "
          f"{g.S_DUAL_2A!r}: difference {d32a!r} ({d32a/g.S_DUAL_2A*100!r} %); 1 % attention line exceeded: {abs(d32a)/g.S_DUAL_2A > 0.01}")
    # --- A4 (1-D): outer minimisation -----------------------------------------------------------------------------------------
    tg = time.time()
    evals, lam_s, Fmin, lo, hi = g.golden(Vlam, EPS, print)
    Sd = 1+Fmin
    print(f"GOLD-1D final bracket [{lo!r}, {hi!r}] (width {hi-lo!r}); eps_lambda {EPS}; {len(evals)} evaluations; wall {time.time()-tg:.1f} s")
    print(f"A4-1D 离散 MDP 对偶估算值 (no acceleration constraint, new grid) S~_dual = 1 + min_lambda {{V~ + {LAM_BUDGET} lambda}} = {Sd!r} "
          f"at lambda* = {lam_s!r} (V~ = {evals[lam_s][0]!r}; v_1 = {evals[lam_s][1]} h); eps_lambda = {EPS}; an estimate, not a bound")
    d32b = Sd-g.S_DUAL_2A
    print(f"3.2(b) outer-minimised new grid {Sd!r} vs old grid {g.S_DUAL_2A!r} (round 2a, eps 1e-3): difference {d32b!r} "
          f"({d32b/g.S_DUAL_2A*100!r} %); 1 % attention line exceeded: {abs(d32b)/g.S_DUAL_2A > 0.01}")
    # --- A6 (1-D): policy at lambda*, forward, T3 ----------------------------------------------------------------------------------
    tp = time.time()
    Vt, l1s, pols = mdp.dual_value(lam_s, c, q, P, phat, Ninv, store=True)
    assert float(Vt) == evals[lam_s][0] and int(l1s) == evals[lam_s][1]
    Sp, ENp, massp, band, low, mk = forward_t3(pols, int(l1s), c, q, P, phat, Ninv)
    Sp0, ENp0, _ = mdp.forward(lambda k: pols[k].astype(np.int64), int(l1s), c, q, P, phat, Ninv)
    print(f"POLICY-1D at lambda* = {lam_s!r}: forward S {Sp!r}, E[N] {ENp!r} (final mass {massp!r}; mdp.forward gives S {Sp0!r}, "
          f"E[N] {ENp0!r}, identical: {Sp == Sp0 and ENp == ENp0}); v_1 = {int(l1s)} h; time {time.time()-tp:.1f} s")
    print(f"POLICY-1D E[N] <= B = {g.B_FULL!r}: {ENp <= g.B_FULL} (E[N] - B = {ENp-g.B_FULL!r})")
    Vchk = (Sp-1)-lam_s*(ENp-1)
    print(f"POLICY-1D diagnostic (not a spec check): V~ {float(Vt)!r} vs (S - 1) - lambda* (E[N] - 1) = {Vchk!r}, relative {abs(Vchk-Vt)/abs(Vt)!r}")
    ks = np.arange(2, K+1)
    T3b, T3l = float(band[ks].mean()), float(low[ks].mean())
    print(f"T3-1D time average over k = 2..{K} (K - 1 = {K-1} steps) of Pr{{18 <= |delta|/sigma <= 19}} = {T3b!r} "
          f"(nodes {g.BAND[0]} <= |i - {g.I0}| <= {g.BAND[1]})")
    print(f"T3-1D time average over k = 2..{K} of Pr{{|v_(k-1)| <= 0.5 sigma}} = {T3l!r} (|l| <= {g.LOW}; v_1 = {int(l1s)} h at k = 2)")
    print(f"T3-1D mass over k: min {float(mk[ks].min())!r} max {float(mk[ks].max())!r}")
    for kk in (500, 1250, 2500, 4000):
        print(f"T3-1D at k = {kk}: Pr{{18 <= |delta|/sigma <= 19}} = {float(band[kk])!r}; Pr{{|v_(k-1)| <= 0.5 sigma}} = {float(low[kk])!r}")
    np.savez_compressed(os.path.join(RES, 'mdp1d_policy.npz'), t=t, pols=pols, l1=int(l1s), lam=lam_s, band=band, low=low)
    out = dict(model='1-D, no acceleration constraint, new grid h = sigma/30, L = 137', eps_lambda=EPS, lam_star=lam_s, lam_star_repr=repr(lam_s),
               V_at_lam_star=float(evals[lam_s][0]), S_dual=Sd, n_evals=len(evals), bracket=[lo, hi], v1=int(l1s),
               check_3_3=dict(S_forward=Sf, S_backward=Sb, EN_forward=ENf, EN_backward=ENb, rel_S=rS, rel_EN=rN, passed=bool(ok33)),
               check_3_2=dict(at_lam_star_2a=dict(lam=g.LAM_STAR_2A, S_new=S2a, S_old=g.S_DUAL_2A, diff=d32a, rel=d32a/g.S_DUAL_2A),
                              outer_min=dict(S_new=Sd, S_old=g.S_DUAL_2A, diff=d32b, rel=d32b/g.S_DUAL_2A)),
               policy=dict(S=Sp, EN=ENp, EN_le_B=bool(ENp <= g.B_FULL), B=g.B_FULL),
               T3=dict(band_18_19=T3b, low_speed=T3l))
    json.dump(out, open(os.path.join(RES, 'mdp1d_result.json'), 'w'), indent=1)
    print(f"=== mdp1d_newgrid.py finished; total wall {time.time()-T0:.1f} s")
