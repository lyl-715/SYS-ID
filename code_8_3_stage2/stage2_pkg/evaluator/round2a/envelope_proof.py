"""Round 2a, B2: certification of the envelope c(delta) <= a q(delta) + b for all real delta, with the fixed coefficients
a = 2.76253, b = 0.094651 (spec reports_CN/plan_recheck_bounds_CN.md v4.2, §B2), hence for every input with
sum_{k>=2} E[q] <= B - 1 <= 316:  S <= 1 + 316 a + 4999 b = 1347.119829 < 1348.

Units. With sigma = Delta/20 (the scenario's definition), t := delta/sigma gives alpha = t - 20 and beta = t + 20 exactly, so the
claim is a statement about the real variable t. Evenness: delta -> -delta maps (alpha, beta) -> (-beta, -alpha), which leaves
P = Phi(beta) - Phi(alpha), c_ev = 1 - P - alpha phi(alpha) + beta phi(beta) and c_no = (phi(alpha) - phi(beta))^2 / P unchanged;
so t >= 0 suffices. g(t) := c - a q - b, q = 1 - P.

Three segments (spec table):
 (i)   0 <= t <= 10: analytic. q = Phi(alpha) + Phi(-beta) <= 2 Phi(-10); |alpha| phi(alpha) <= 10 phi(10) and beta phi(beta)
       <= 10 phi(10) since x phi(x) decreases for x >= 1; so c_ev <= 2 Phi(-10) + 20 phi(10). P >= Phi(20) - Phi(-10)
       >= Phi(10) - Phi(-10), and c_no <= (phi(alpha) + phi(beta))^2 / P <= (2 phi(10))^2 / P. The constants are evaluated in Arb
       ball arithmetic (rigorous enclosures) and compared: c < 1e-20 < b, and a q >= 0, so g < 0.
 (ii)  10 <= t <= 20: interval arithmetic on the formula, in Arb (python-flint; rigorous ball arithmetic, outward rounding;
       mpmath.iv has no working erf/erfc/ncdf in this environment): Phi(x) = erfc(-x/sqrt 2)/2, phi(x) = exp(-x^2/2)/sqrt(2 pi),
       P = Phi(beta) - Phi(alpha), q = 1 - P, and
           g = (1 - a) q - alpha phi(alpha) + beta phi(beta) + (phi(alpha) - phi(beta))^2 / P - b,
       which is c_ev + c_no - a q - b with the two occurrences of q collected. [10, 20] is cut at the exact rationals 10 + i/100
       (1000 segments of width 0.01); each segment is the Arb ball enclosing [t0, t1] (union of the balls of the two rational
       endpoints). A segment is certified when the whole enclosure of g is < 0. Otherwise it is bisected (at exact rational
       midpoints) while the children are at least 1e-6 wide, i.e. down to width 0.01/2^13 = 1.22e-6; a piece still not
       certified there is "未认证" and the bound is "候选系数未通过认证". P >= Phi(30) - 1/2 > 0.49 here (checked), so 1/P is safe.
 (iii) t >= 20 (alpha >= 0): analytic. q >= Phi(alpha) >= 1/2 and c <= 1, so a q + b >= a/2 + b = 1.475916 > 1 >= c.
       c <= 1: with u = (z - x_k)/sigma ~ N(0, 1) and the no-send cell A = {alpha < u < beta}, c_ev = E[u^2 1_{not A}] and
       c_no = E[u 1_A]^2 / P(A), so c = 1 - E[u^2 1_A] + E[u 1_A]^2 / P(A) <= 1 by Cauchy-Schwarz (E[u 1_A]^2 <= E[u^2 1_A] P(A)).

mc83.cfun is not used in the certificate; it is evaluated at a few points afterwards as a sanity cross-check only.
Writes logs/envelope_proof.log (one line per segment and per number), results/envelope_segments.json.
Usage (from this directory): python3 envelope_proof.py"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE); sys.path.insert(0, EV)
from flint import arb, fmpq, ctx

ctx.prec = 256
A_Q = fmpq(276253, 100000); B_Q = fmpq(94651, 1000000)          # a = 2.76253, b = 0.094651 (exact)
A, B = arb(A_Q), arb(B_Q)
SQRT2 = arb(2).sqrt(); SQRT2PI = (2*arb.pi()).sqrt()
MIN_WIDTH = fmpq(1, 1000000); MAX_DEPTH = 13                    # 0.01/2^13 = 1.22e-6 >= 1e-6 > 0.01/2^14
LOGS = os.path.join(HERE, 'logs'); RES = os.path.join(HERE, 'results')


def Phi(x): return (-x/SQRT2).erfc()/2
def phi(x): return (-(x*x)/2).exp()/SQRT2PI

def g_ball(T):
    """enclosure of g over the ball T (t = delta/sigma), and of P (to show the 1/P safety)."""
    al = T-20; be = T+20
    P = Phi(be)-Phi(al); q = 1-P
    pa, pb = phi(al), phi(be)
    d = pa-pb
    return (1-A)*q - al*pa + be*pb + d*d/P - B, P                  # squares as products (rigorous for any ball)

def ball(t0, t1): return arb(t0).union(arb(t1))

def certify(t0, t1, depth, stats):
    """True if g < 0 is certified on [t0, t1] (bisecting as allowed); records leaves (t0, t1, upper) in stats."""
    G, P = g_ball(ball(t0, t1))
    if not (P > arb('0.49')): raise RuntimeError(f'P enclosure not > 0.49 on [{t0}, {t1}]')
    stats['max_depth'] = max(stats.get('max_depth', 0), depth)
    if G < 0:
        stats['leaves'].append((t0, t1, G.upper())); return True
    if depth == MAX_DEPTH:
        stats['leaves'].append((t0, t1, G.upper())); stats['uncertified'].append((t0, t1, G.upper())); return False
    stats['bisections'] += 1
    tm = (t0+t1)/2
    ok1 = certify(t0, tm, depth+1, stats); ok2 = certify(tm, t1, depth+1, stats)
    return ok1 and ok2

def fmt(x): return x.str(12, radius=False)


if __name__ == '__main__':
    os.makedirs(LOGS, exist_ok=True); os.makedirs(RES, exist_ok=True)
    import density_recursion_v1 as dr
    sys.stdout = dr.Tee(os.path.join(LOGS, 'envelope_proof.log'))
    t_start = time.time()
    S_bar = 1+316*A_Q+4999*B_Q
    print(f"=== envelope_proof.py (round 2a B2); spec v4.2 commit 6863d7c; Arb (python-flint) precision {ctx.prec} bits")
    print(f"ENV coefficients a = {A_Q} = 2.76253, b = {B_Q} = 0.094651 (exact rationals); S_bar = 1 + 316 a + 4999 b = {S_bar} "
          f"= {float(S_bar):.6f} (exact rational)")
    # (i) 0 <= t <= 10
    ub_cev = 2*Phi(arb(-10)) + 20*phi(arb(10)); lb_P = Phi(arb(10))-Phi(arb(-10)); ub_cno = (2*phi(arb(10)))*(2*phi(arb(10)))/lb_P
    ok_i = bool(ub_cev < arb('1e-20') and lb_P > arb('0.99') and ub_cev+ub_cno < arb('1e-20') and arb('1e-20') < B)
    print(f"ENV (i) 0 <= delta <= Delta - 10 sigma: c_ev <= 2 Phi(-10) + 20 phi(10) = {fmt(ub_cev)} (< 1e-20: {ub_cev < arb('1e-20')}); "
          f"P >= Phi(10) - Phi(-10) = {fmt(lb_P)} (> 0.99: {lb_P > arb('0.99')}); c_no <= (2 phi(10))^2/P <= {fmt(ub_cno)}; "
          f"c <= {fmt(ub_cev+ub_cno)} < 1e-20 < b; g < 0: {ok_i}")
    # (iii) t >= 20
    lb_iii = A_Q/2+B_Q; ok_iii = bool(lb_iii > 1)
    print(f"ENV (iii) delta >= Delta: q >= Phi(alpha) >= 1/2, c <= 1 (Cauchy-Schwarz, see docstring); a q + b >= a/2 + b = {lb_iii} "
          f"= {float(lb_iii)} > 1: {ok_iii}")
    # (ii) 10 <= t <= 20
    seg = []; tot_bis = 0; unc = []; worst = None; t_ii = time.time(); maxd = 0; minw = None
    print(f"ENV (ii) on [10, 20]: beta >= 30 and alpha <= 0, so P >= Phi(30) - 1/2 = {fmt(Phi(arb(30))-arb('0.5'))} > 0.49; "
          f"the enclosure of P on every segment and piece is also checked to be > 0.49 (else the script stops)")
    for i in range(1000):
        t0 = fmpq(1000+i, 100); t1 = fmpq(1001+i, 100)
        G0, _ = g_ball(ball(t0, t1))
        st = dict(leaves=[], uncertified=[], bisections=0)
        ok = certify(t0, t1, 0, st)
        up = max((u for _, _, u in st['leaves']), key=lambda u: float(u.mid()))
        tot_bis += st['bisections']; unc += st['uncertified']; maxd = max(maxd, st.get('max_depth', 0))
        w_ = min(b_-a_ for a_, b_, _ in st['leaves']); minw = w_ if minw is None or w_ < minw else minw
        if worst is None or float(up.mid()) > float(worst[2].mid()): worst = (t0, t1, up)
        seg.append(dict(t0=str(t0), t1=str(t1), upper_initial=fmt(G0.upper()), bisections=st['bisections'], leaves=len(st['leaves']),
                        max_leaf_upper=fmt(up), certified=ok))
        print(f"SEG {i:4d} t in [{float(t0):.2f}, {float(t1):.2f}] upper(g) initial {fmt(G0.upper())}; bisections {st['bisections']}; "
              f"leaves {len(st['leaves'])}; max leaf upper {fmt(up)}; {'certified' if ok else '未认证'}")
    print(f"ENV (ii) segments 1000; total bisections {tot_bis}; uncertified pieces {len(unc)}; worst (largest) leaf upper endpoint "
          f"{fmt(worst[2])} in segment [{float(worst[0]):.2f}, {float(worst[1]):.2f}]; deepest bisection level {maxd} (limit {MAX_DEPTH}); "
          f"narrowest piece {minw} = {float(minw):.4e} (>= 1e-6: {minw >= MIN_WIDTH}); time {time.time()-t_ii:.1f} s")
    ok_ii = not unc
    for t0, t1, u in unc: print(f"ENV 未认证 piece [{float(t0)!r}, {float(t1)!r}] upper {fmt(u)}")
    # sanity cross-check against mc83.cfun (float), not part of the certificate
    import numpy as np
    for t in (0.0, 5.0, 10.0, 15.0, 18.47, 19.5, 20.0, 25.0):
        al = np.array([t-20.0]); be = np.array([t+20.0]); cev, cno = dr.cfun(al, be)
        Ta = arb(fmpq(int(round(t*100)), 100)); ala, bea = Ta-20, Ta+20; Pa = Phi(bea)-Phi(ala); qa = Phi(ala)+Phi(-bea)
        da = phi(ala)-phi(bea); ca = qa-ala*phi(ala)+bea*phi(bea)+da*da/Pa      # 1 - P written as Phi(alpha) + Phi(-beta) (no cancellation)
        print(f"XCHECK t = {t}: cfun c = {float(cev[0]+cno[0])!r}, Arb c = {fmt(ca)}, |difference| = {abs(float(cev[0]+cno[0])-float(ca.mid())):.2e}")
    passed = ok_i and ok_ii and ok_iii
    print(f"ENV total runtime {time.time()-t_start:.1f} s")
    print(f"ENV VERDICT: {'PASS: c <= a q + b certified for all real delta; S <= 1347.119829 < 1348 (解析认证的上界)' if passed else '候选系数未通过认证'}")
    json.dump(dict(a=str(A_Q), b=str(B_Q), S_bar=str(S_bar), segment_i=ok_i, segment_iii=ok_iii, segments_ii=seg, total_bisections=tot_bis,
                   uncertified=[(str(a), str(b), fmt(u)) for a, b, u in unc], worst=(str(worst[0]), str(worst[1]), fmt(worst[2])), passed=passed),
              open(os.path.join(RES, 'envelope_segments.json'), 'w'), indent=1, ensure_ascii=False)
