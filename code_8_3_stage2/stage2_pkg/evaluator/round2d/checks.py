"""Round 2d, Part B: the checks that must pass before the outer loop (task B1, B2; spec reports_CN/plan_round2d_CN.md §3 items 1-2).

B1  regression: at (A~, lambda) = (I, lambda*_2c), lambda*_2c = 0.020120725733454057 (round2c/results/outer_2d_A1.json, archived full
    precision, 2-D A_1 case), D - 197.197 reproduces round 2c's 4.385285228783914 to relative 1e-10. Logged next to it: V~ against
    round 2c's 6.2948236330331815, gamma(I) against round2c/results/gamma_A1.npy (bitwise), and the trace identity
    sum_k w_k(u_hat) gamma_k(I) = 9 (round 2c B1) as diagnostics.
B2  subgradient validity (supporting hyperplane, the definition of a subgradient of a convex function; no finite differences):
    at A~ = I, lambda = lambda*_2c, with the reported G_A~ and G_lambda of the oracle,
      D(I + eps Delta, lambda) >= D(I, lambda) + eps tr(G_A~ Delta) - 1e-9   for 3 random symmetric Delta, ||Delta||_F = 1,
                                                                           eps in {1e-2, 1e-3};
      D(I, lambda +- eps) >= D(I, lambda) +- eps G_lambda - 1e-9            for eps = 1e-4.
    The slack is also split into its log-det part -log det(I + eps Delta) + eps tr(Delta) (>= 0, ~ eps^2/2) and the rest
    (gamma_1 + V~ part, >= 0 by convexity of V~), both reported.
Writes logs/checks.log and results/checks.json (all_passed gates logdet_outer.py outer). Usage (from this directory):
    setsid nohup python3 checks.py --threads 4 > logs/checks.stdout 2>&1 &"""
import os, sys, json, time, argparse
import numpy as np
import logdet_outer as lo
from logdet_outer import g, mdp, dr, K, NPAR, BM1, LD_JC, EYE, LOGS, RES
from numba import set_num_threads, get_num_threads

TOL_REL = 1e-10; TOL_SUP = 1e-9
SEED = 20261022                              # rng seed of the three directions (spec date; DECISION B-1)
EPS_A = (1e-2, 1e-3); EPS_L = 1e-4


def b1(model, log):
    gA1 = np.load(os.path.join(lo.R2C, 'results', 'gamma_A1.npy')); gI = lo.gamma_of(EYE)
    wh = np.load(os.path.join(lo.R2C, 'results', 'w_NM-none-P5514.npy'))
    tr9 = float(wh@gI); Gh = lo.gram(wh, 1)
    log(f"B1 setup: M_0 = round2c/results/M_A1.npy; log det M_0 (Cholesky) = {lo.LDM0!r}; gamma(I) bitwise equal to round2c/results/gamma_A1.npy: "
        f"{np.array_equal(gI, gA1)} (max |diff| {float(np.abs(gI-gA1).max())!r}); diagnostic sum_k w_k(u_hat) gamma_k(I) = {tr9!r} (9); "
        f"max |sum_k w_k(u_hat) g_k g_k^T - I| = {float(np.abs(Gh-EYE).max())!r}")
    o = lo.oracle(EYE, lo.LAM_2C, model); dm = o['D']-LD_JC; rel = abs(dm-lo.DM_2C)/abs(lo.DM_2C); ok = rel <= TOL_REL
    log(f"B1 oracle at (I, lambda*_2c = {lo.LAM_2C!r}): log det M_0 - log det I - 9 + gamma_1 = {o['const']!r} (round 2c const {lo.OUT2C['const']!r}); "
        f"V~ = {o['V']!r} (round 2c {lo.V_2C!r}; bitwise equal {o['V'] == lo.V_2C}; v_1 = {o['l1']} h); D = {o['D']!r}; DP with policy storage "
        f"{o['t_dp']:.1f} s, policy mismatches {o['bad']}; numba threads {get_num_threads()}")
    log(f"B1 D - 197.197 = {dm!r} vs round 2c {lo.DM_2C!r} (round2c/results/outer_2d_A1.json): relative difference {rel!r} (<= 1e-10: {ok}; "
        f"bitwise equal: {dm == lo.DM_2C})")
    log(f"B1 forward of the stored policy ({o['t_fwd']:.1f} s): n = sum_(k>=2) E[q] = {o['n']!r}; sum_(k>=2) gamma_k E[c] - lambda n = {o['fwd_value']!r} "
        f"vs V~: relative {o['fwd_rel']!r} (diagnostic); mass at K {o['mass_K']!r}; G_lambda = (B - 1) - n = {o['Glam']!r}; ||G_A~||_F = "
        f"{float(np.linalg.norm(o['GA']))!r}; tr G_A~ = {float(np.trace(o['GA']))!r}")
    return o, dict(D=o['D'], D_minus_LDJC=dm, ref=lo.DM_2C, rel=rel, bitwise=bool(dm == lo.DM_2C), V=o['V'], V_2c=lo.V_2C,
                   gamma_bitwise=bool(np.array_equal(gI, gA1)), trace_identity=tr9, n=o['n'], G_lam=o['Glam'], fwd_rel=o['fwd_rel'],
                   bad=o['bad'], t_dp=o['t_dp'], t_fwd=o['t_fwd'], passed=bool(ok))

def b2(o, model, log):
    lam = lo.LAM_2C; D0 = o['D']; GA = o['GA']; rng = np.random.default_rng(SEED); rows = []; ok = True
    for d in range(3):
        Z = rng.standard_normal((NPAR, NPAR)); Dl = 0.5*(Z+Z.T); Dl /= np.linalg.norm(Dl)
        tg = float(np.sum(GA*Dl))
        for eps in EPS_A:
            X = EYE+eps*Dl; De, Ve, l1e, tdp = lo.D_value(X, lam, model)
            lin = D0+eps*tg; s = De-lin; good = s >= -TOL_SUP
            sld = -lo.logdet_pd(X)[1]+eps*float(np.trace(Dl)); srest = s-sld
            log(f"B2 direction {d+1} (||Delta||_F = {float(np.linalg.norm(Dl))!r}, tr Delta = {float(np.trace(Dl))!r}), eps = {eps:g}: "
                f"D(I + eps Delta) = {De!r}; D(I) + eps tr(G Delta) = {lin!r} (tr(G Delta) = {tg!r}); slack = {s!r} (>= -1e-9: {good}); "
                f"log-det part {sld!r}, gamma_1 + V~ part {srest!r}; V~ = {Ve!r} (v_1 = {l1e} h; {tdp:.1f} s)")
            rows.append(dict(kind='A', direction=d+1, eps=eps, D=De, lin=lin, slack=s, slack_logdet=sld, slack_rest=srest, V=Ve, passed=bool(good)))
            ok &= good
    for sgn in (+1, -1):
        lm = lam+sgn*EPS_L; De, Ve, l1e, tdp = lo.D_value(EYE, lm, model)
        lin = D0+sgn*EPS_L*o['Glam']; s = De-lin; good = s >= -TOL_SUP
        log(f"B2 lambda {'+' if sgn > 0 else '-'} 1e-4 = {lm!r}: D = {De!r}; D(I, lambda) {'+' if sgn > 0 else '-'} 1e-4 G_lambda = {lin!r}; "
            f"slack = {s!r} (>= -1e-9: {good}); V~ = {Ve!r} (v_1 = {l1e} h; {tdp:.1f} s)")
        rows.append(dict(kind='lambda', sign=sgn, eps=EPS_L, lam=lm, D=De, lin=lin, slack=s, V=Ve, passed=bool(good))); ok &= good
    mn = min(r['slack'] for r in rows)
    log(f"B2 smallest slack over the {len(rows)} tests = {mn!r}; all >= -1e-9: {ok}")
    return dict(seed=SEED, rows=rows, min_slack=mn, passed=bool(ok))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--threads', type=int, default=None); args = ap.parse_args()
    if args.threads: set_num_threads(args.threads)
    sys.stdout = dr.Tee(os.path.join(LOGS, 'checks.log'), 'a')
    def log(m): print(m, flush=True)
    T0 = time.time()
    log(f"\n=== checks.py started {time.strftime('%Y-%m-%d %H:%M:%S')} (round 2d Part B); numba threads {get_num_threads()}; load {g.loadavg()}")
    model = mdp.model()
    o, r1 = b1(model, log)
    r2 = b2(o, model, log)
    allp = bool(r1['passed'] and r2['passed'])
    log(f"B SUMMARY B1 {r1['passed']}, B2 {r2['passed']}; all passed: {allp}")
    json.dump(dict(spec='reports_CN/plan_round2d_CN.md (ac058c4)', B1=r1, B2=r2, all_passed=allp), open(os.path.join(RES, 'checks.json'), 'w'), indent=1)
    if not allp: log("CHECKS FAILED: STOPPED (the outer loop refuses to run)")
    log(f"=== checks.py finished {time.strftime('%Y-%m-%d %H:%M:%S')}; wall {time.time()-T0:.1f} s")
