"""Round 2c, Part B: the four checks of spec reports_CN/plan_round2c_CN.md §3 (all must pass before any estimate is reported).

B1  trace identity: sum_k w_k(u_hat) gamma^(1)_k = 9 and sum_k w_U,k gamma^(2)_k = 9, relative difference <= 1e-10.
B2  inequalities at known inputs (U and the seven round-2a candidates), archived J(u) (round2a/results/J_<id>.npy) and w(u)
    (results/w_<id>.npy, Part A): log det J(u) <= log det M(w(u)) <= tr(A_i M(w(u))) - log det A_i - 9, i = 1, 2.
    slack1 = log det M(w(u)) - log det J(u); slack2_i = tr(A_i M(w(u))) - log det A_i - 9 - log det M(w(u)), with
    tr(A_i M(w(u))) = sum_k w_k gamma^(i)_k (the per-step form the DP uses; tr(cho_solve(L_i, M(w(u)))) logged as a diagnostic).
    An inequality counts as holding when its slack >= -1e-10 nat (rounding of log det ~ 197: DECISION B-1); for P5514 with A_1,
    |slack2| <= 1e-8.
B3  regression to the S route: gamma_k = 1 for all k (first step 1), 2-D DP at the archived round-2b lambda* = 0.9381038031641779
    reproduces V~ = 321.78972743542596 (round2b/logs/gold_2d.jsonl:21), relative <= 1e-10; S~_acc = 1 + V~ + 315.892 lambda* vs
    619.1292140245644. Extra (same tolerance, not a spec check): the 1-D DP at the round-2b 1-D lambda* vs round2b/logs/gold_1d.jsonl:17.
B4  forward-backward consistency with the gamma-weighted reward: the uniform input's increments rounded to multiples of h as a
    fixed open-loop policy (round 2b's 3.3 policy, mdp2d_estimate.open_loop_codes); forward sum_{k=2..K} gamma_k E[c(delta_k)] vs
    the fixed-policy backward value, relative <= 1e-10, 2-D model, gamma^(1) and gamma^(2). Extra: the same in the 1-D model.
Writes logs/part_b.log and results/checks.json (all_passed gates Part C). Usage (from this directory):
    nohup python3 checks.py all --threads 4 > logs/checks.stdout 2>&1 &"""
import os, sys, json, time, argparse
import numpy as np
from scipy.linalg import cho_solve
import logdet_dual as ld
from logdet_dual import g, mdp, m2, dr, K, LOGS, RES, EV, IDS, UHAT, NPAR, BM1
from numba import set_num_threads, get_num_threads

TOL_REL = 1e-10; TOL_INEQ = -1e-10; TOL_EQ = 1e-8


def b1(log):
    out = {}; ok = True
    for i, src in ld.SRC.items():
        w = np.load(os.path.join(RES, f'w_{src}.npy')); Mi, Lc, ldM, A, gam = ld.load_A(i)
        s = float(w@gam); rel = abs(s-NPAR)/NPAR; tr = float(np.trace(A@Mi))
        log(f"B1 A_{i} with w({src}): sum_k w_k gamma^({i})_k = {s!r}; relative difference to 9 = {rel!r} (<= 1e-10: {rel <= TOL_REL}); "
            f"diagnostic tr(A_{i} M_{i}) with the stored A_{i} = {tr!r}")
        out[f'A{i}'] = dict(source=src, sum_w_gamma=s, rel=rel, passed=bool(rel <= TOL_REL), trace_AM_stored=tr); ok &= rel <= TOL_REL
    out['passed'] = bool(ok); return out

def b2(log):
    ld.register_candidates(); HU = ld.h_uniform(); As = {i: ld.load_A(i) for i in (1, 2)}; rows = {}; ok = True
    for i in IDS:
        a, chi = ld.ac_of(i)
        J = np.load(os.path.join(EV, 'round2a', 'results', f'J_{i}.npy')); w = np.load(os.path.join(RES, f'w_{i}.npy'))
        Mw = ld.M_of(HU, w); ldJ = float(np.linalg.slogdet(J)[1]); ldMw = float(np.linalg.slogdet(Mw)[1])
        s1 = ldMw-ldJ; pw = -2*float(np.log(4*a**2).sum())
        r = dict(logdet_J=ldJ, logdet_M=ldMw, slack1=s1, slack1_power_term=pw, logdet_J_recheck7=ld.R7[i]['logdet'])
        good = s1 >= TOL_INEQ
        line = (f"B2 {i}: log det J(u) {ldJ!r} (recheck7 {ld.R7[i]['logdet']!r}); log det M(w(u)) {ldMw!r}; slack1 = {s1!r} "
                f"(-2 sum log(4 p_m) = {pw!r}; holds: {s1 >= TOL_INEQ})")
        for k_, (Mi, Lc, ldA_M, A, gam) in As.items():
            tr = float(w@gam); tr2 = float(np.trace(cho_solve((Lc, True), Mw))); rhs = tr+ldA_M-NPAR; s2 = rhs-ldMw
            r[f'A{k_}'] = dict(trace=tr, trace_cho=tr2, rhs=rhs, slack2=s2, holds=bool(s2 >= TOL_INEQ)); good &= s2 >= TOL_INEQ
            line += (f"; A_{k_}: tr(A M(w(u))) = sum w gamma = {tr!r} (cho_solve {tr2!r}), tr - log det A - 9 = {rhs!r}, slack2 = {s2!r} "
                     f"(holds: {s2 >= TOL_INEQ})")
        log(line); r['holds'] = bool(good); rows[i] = r; ok &= good
    s2h = rows[UHAT]['A1']['slack2']; eq = abs(s2h) <= TOL_EQ
    log(f"B2 {UHAT} with A_1: slack2 = {s2h!r}; |slack2| <= 1e-8: {eq}")
    log("B2 table | input | slack1 | slack2 (A_1) | slack2 (A_2) |")
    for i in IDS: log(f"B2 table | {i} | {rows[i]['slack1']!r} | {rows[i]['A1']['slack2']!r} | {rows[i]['A2']['slack2']!r} |")
    return dict(rows=rows, equality_P5514_A1=dict(slack2=s2h, passed=bool(eq)), all_hold=bool(ok), passed=bool(ok and eq))

def b3(model, log):
    one = np.ones(K)
    tb = time.time(); V2, l12 = ld.dp2d_gamma(ld.LAM_2B, one, model); t2 = time.time()-tb
    rel = abs(V2-ld.V_2B)/abs(ld.V_2B); S2 = 1.0+V2+BM1*ld.LAM_2B; relS = abs(S2-ld.S_2B)/ld.S_2B
    log(f"B3 2-D, gamma_k = 1, lambda = {ld.LAM_2B!r}: V~ {V2!r} (v_1 = {l12} h; {t2:.1f} s, numba threads {get_num_threads()}) vs round 2b {ld.V_2B!r}: "
        f"relative difference {rel!r} (<= 1e-10: {rel <= TOL_REL}; bitwise equal: {V2 == ld.V_2B}); 1 + V~ + {BM1} lambda = {S2!r} vs "
        f"S~_acc = {ld.S_2B!r}: relative {relS!r}")
    tb = time.time(); V1, l11 = ld.dp1d_gamma(ld.LAM_1D_2B, one, model); t1 = time.time()-tb
    rel1 = abs(V1-ld.V_1D_2B)/abs(ld.V_1D_2B)
    log(f"B3 extra (not a spec check): 1-D, gamma_k = 1, lambda = {ld.LAM_1D_2B!r}: V~ {V1!r} (v_1 = {l11} h; {t1:.1f} s) vs round 2b {ld.V_1D_2B!r} "
        f"(round2b/logs/gold_1d.jsonl:17): relative {rel1!r} (<= 1e-10: {rel1 <= TOL_REL}; bitwise equal: {V1 == ld.V_1D_2B})")
    return dict(V=V2, V_2b=ld.V_2B, rel=rel, S=S2, S_2b=ld.S_2B, rel_S=relS, l1=l12, time_s=t2, passed=bool(rel <= TOL_REL),
                extra_1d=dict(lam=ld.LAM_1D_2B, V=V1, V_2b=ld.V_1D_2B, rel=rel1, passed=bool(rel1 <= TOL_REL), time_s=t1))

def b4(model, log):
    t, c, q, P, phat, Ninv, N = model
    lfull, rerr, inc = g.uniform_open_loop(); l1 = int(lfull[0]); dl = int(np.abs(np.diff(lfull)).max())
    log(f"B4 open-loop policy: the uniform input's increments rounded to multiples of h (round 2b 3.3): |l| max {int(np.abs(lfull).max())}; "
        f"max rounding error {rerr!r} h; max |l_(k+1) - l_k| = {dl}; v_1 = {l1} h")
    at = m2.open_loop_codes(lfull); tb = time.time()
    Ec, Eq, mass = ld.forward2d_steps(at, l1, model); tf = time.time()-tb
    S = 1.0
    for k in range(2, K+1): S += float(Ec[k])
    log(f"B4 2-D forward ({tf:.1f} s): S = 1 + sum_k E[c] = {S!r} (round 2b 3.3 forward 466.3091828526577); mass at K {float(mass[K])!r}")
    out = dict(l1=l1, S_forward=S, models={}); ok = True
    lseq = np.zeros(K+1, np.int64); lseq[2:K] = lfull[1:K-1]
    Ec1, Eq1 = ld.forward1d_steps(lseq, l1, model)
    for i in (1, 2):
        gam = ld.load_A(i)[4]
        fw = 0.0
        for k in range(2, K+1): fw += float(gam[k-1]*Ec[k])
        tb = time.time(); bw = ld.backward_fixed2d_gamma(at, l1, gam, model); tbw = time.time()-tb
        rel = abs(fw-bw)/abs(bw)
        log(f"B4 2-D gamma^({i}): forward sum_(k=2..K) gamma_k E[c] = {fw!r}; fixed-policy backward = {bw!r} ({tbw:.1f} s); relative difference "
            f"{rel!r} (<= 1e-10: {rel <= TOL_REL}); with gamma_1: {float(gam[0])+fw!r}")
        fw1 = 0.0
        for k in range(2, K+1): fw1 += float(gam[k-1]*Ec1[k])
        tb = time.time(); bw1 = ld.backward_fixed1d_gamma(lseq, l1, gam, model); tb1 = time.time()-tb
        rel1 = abs(fw1-bw1)/abs(bw1)
        log(f"B4 extra (not a spec check) 1-D gamma^({i}): forward {fw1!r}; backward {bw1!r} ({tb1:.1f} s); relative {rel1!r} (<= 1e-10: {rel1 <= TOL_REL}); "
            f"2-D - 1-D forward {fw-fw1!r}")
        out['models'][f'A{i}'] = dict(forward_2d=fw, backward_2d=bw, rel_2d=rel, passed=bool(rel <= TOL_REL), forward_1d=fw1, backward_1d=bw1,
                                      rel_1d=rel1, passed_1d=bool(rel1 <= TOL_REL))
        ok &= rel <= TOL_REL
    out['passed'] = bool(ok); return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['all']); ap.add_argument('--threads', type=int, default=None)
    args = ap.parse_args()
    if args.threads: set_num_threads(args.threads)
    sys.stdout = dr.Tee(os.path.join(LOGS, 'part_b.log'), 'a')
    def log(m): print(m, flush=True)
    T0 = time.time()
    log(f"\n=== checks.py {args.stage} started {time.strftime('%Y-%m-%d %H:%M:%S')} (round 2c Part B); numba threads {get_num_threads()}; load {g.loadavg()}")
    res = dict(B1=b1(log), B2=b2(log))
    model = mdp.model()
    res['B3'] = b3(model, log)
    res['B4'] = b4(model, log)
    allp = all(res[k]['passed'] for k in ('B1', 'B2', 'B3', 'B4'))
    res['all_passed'] = bool(allp)
    log(f"B SUMMARY B1 {res['B1']['passed']}, B2 {res['B2']['passed']}, B3 {res['B3']['passed']}, B4 {res['B4']['passed']}; all passed: {allp}"
        + ("" if allp else " -> STOPPED: no estimate of Part C is reported"))
    json.dump(res, open(os.path.join(RES, 'checks.json'), 'w'), indent=1)
    log(f"=== checks.py finished {time.strftime('%Y-%m-%d %H:%M:%S')}; wall {time.time()-T0:.1f} s")
