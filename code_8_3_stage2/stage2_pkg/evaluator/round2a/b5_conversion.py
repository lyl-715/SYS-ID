"""Round 2a, B5: conversion of the S bound into log det J (spec reports_CN/plan_recheck_bounds_CN.md v4.2, §B5).

1. Finding on the route-1 certificate (read from code_8_3_stage2/stage2_pkg/bound/bound_common.py and bound_report.md, logged):
   whether its upper value U^upper (cutting-plane bound U_cp) includes conservative floating-point handling.
2. U^upper(S; u_unif) with the route-1 solver, unchanged (bound_common.py exec'd in its own namespace; nothing is written to
   bound/): exchange(H_u, W, cap = 1, tol = 1e-10, batch = 256) and certificate_full, at W = 1347.119829 (the certified S bound,
   B <= 317), and as checks at W = 465.957628 (reproducing bound_inner.log:31) and at the MDP estimate W = S~_dual.
3. The B5 case: (1) envelope passed and U^upper conservative; (2) passed, not conservative; (3) envelope not passed.
4. Estimates (all cases): U^upper - 197.197 and U(S~_dual) - 197.197 as estimates of the relative gain, never relative bounds.
Writes logs/b5_conversion.log. Usage (after envelope_proof.py and mdp_estimate.py): python3 b5_conversion.py"""
import os, sys, json, re, time
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE); PKG = os.path.dirname(EV)
sys.path.insert(0, EV)
import density_recursion_v1 as dr

S_BAR = 1347.119829                     # 1 + 316 a + 4999 b (exact rational 1347119829/1000000), logs/envelope_proof.log
LD_REF = 197.197                        # spec §B5


def bound_namespace():
    """exec bound_common.py unchanged in its own namespace; restore the working directory afterwards."""
    cwd = os.getcwd(); ns = {'__file__': os.path.join(PKG, 'bound', 'bound_common.py')}
    exec(open(ns['__file__']).read(), ns); os.chdir(cwd)
    return ns

def solve(ns, W):
    t0 = time.time(); rx = ns['exchange'](ns['H_u'], W, 1.0, tol=1e-10, batch=256)
    cx = ns['certificate_full'](ns['H_u'], rx['xi'], W, 1.0)
    return cx, rx, time.time()-t0


if __name__ == '__main__':
    sys.stdout = dr.Tee(os.path.join(HERE, 'logs', 'b5_conversion.log'))
    print("=== b5_conversion.py (round 2a B5); spec v4.2 commit 6863d7c")
    src = open(os.path.join(PKG, 'bound', 'bound_common.py')).read().splitlines()
    ln = {k: next(i for i, l in enumerate(src, 1) if l.startswith(k)) for k in ('def ld_grad', 'def top_sum', 'def certificate_full')}
    rep = open(os.path.join(PKG, 'bound', 'bound_report.md')).read()
    kw = [w for w in ('interval', 'rounding error', 'floating-point', 'outward', 'rigorous') if w in rep.lower()]
    print(f"B5 finding: the route-1 certificate U^upper = U_cp = F(xi~) + top_sum(d) - xi~^T d (bound_common.py:{ln['def certificate_full']}) "
          f"is computed in plain float64 numpy: F and d from Cholesky / triangular solve / log of the diagonal (bound_common.py:{ln['def ld_grad']}), "
          f"top_sum by sorting and summation (bound_common.py:{ln['def top_sum']}); no interval arithmetic, outward rounding or "
          f"rounding-error term is added. bound_report.md mentions none of these ({kw if kw else 'no keyword found'}). "
          f"=> U^upper is NOT floating-point conservative.")
    env = json.load(open(os.path.join(HERE, 'results', 'envelope_segments.json')))
    print(f"B5 envelope proof passed (results/envelope_segments.json): {env['passed']}")
    ns = bound_namespace()
    ns['exchange'](ns['H_u'][::25], 50.0, 1.0, tol=1e-3, max_sweeps=5)        # numba compile
    mdp = open(os.path.join(HERE, 'logs', 'mdp_estimate.log')).read()
    S_dual = float(re.search(r"S~_dual = 1 \+ min_lambda \{V~ \+ 315\.892 lambda\} = ([0-9.e+-]+)", mdp).group(1))
    U = {}
    for lab, W in (('check: S(u_unif) of route 1 (bound_inner.log:31)', ns['W_B']), ('certified S bound', S_BAR),
                   ('MDP estimate S~_dual', S_dual)):
        cx, rx, tt = solve(ns, W); U[lab] = cx['U_cp']
        print(f"B5 U at W = {W!r} ({lab}): F(xi~) = {cx['F']!r}, U^upper (cutting plane) = {cx['U_cp']!r}, gap {cx['gap_cp']!r}; "
              f"duality bound {float(cx['U_dual'])!r}; {rx['sweeps']} sweeps, {tt:.1f} s")
    Uu, Ub, Ud = U['check: S(u_unif) of route 1 (bound_inner.log:31)'], U['certified S bound'], U['MDP estimate S~_dual']
    print(f"B5 check: U(465.957628) = {Uu!r} vs bound_inner.log:31 198.4234663321 (difference {Uu-198.4234663321:+.2e})")
    case = 3 if not env['passed'] else 2      # the certificate is not floating-point conservative (finding above)
    print(f"B5 CASE {case}: " + {2: "envelope proof passed, U^upper not floating-point conservative -> the only rigorous statement is "
                                    "S <= 1347.119829; the log det number is a 数值证书，未作浮点保守处理, and no claim is made that the original "
                                    "model satisfies it",
                                 3: "envelope proof not passed -> no certified S bound and no rigorous log det statement this round"}[case])
    print(f"B5 log det number at S = 1347.119829: U^upper = {Ub!r} (数值证书，未作浮点保守处理)")
    print(f"B5 estimate (not a bound): U^upper - {LD_REF} = {Ub-LD_REF!r} nat (relative-gain estimate; no certified lower bound on log det J_C; "
          f"{LD_REF} as in spec §B5, = log det J_sod(u_unif) of the R = 10000 weights, bound_common LD_SOD {ns['LD_SOD']!r}; "
          f"the evaluator B/n=24 gives 197.20097668224076)")
    print(f"B5 estimate (not a bound): MDP analogue U(S~_dual = {S_dual!r}) - {LD_REF} = {Ud-LD_REF!r} nat")
    print(f"B5 B6 comparison: U^upper(1347.12) - U(465.958) = {Ub-Uu!r} nat; 9 log(1347/466) = {9*__import__('math').log(1347/466)!r}")
