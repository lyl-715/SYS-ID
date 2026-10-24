"""Round 5, Part C: headline numbers by the rules of spec §4 and the predictions R1-R6 against the outcomes (spec
reports_CN/plan_round5_CN.md v3.2, frozen; readings in decisions_round5.md, DECISION S-1 .. S-8). Reads only the archives of
Parts A and B; no evaluation. Round 6 Part 0a: 非单调 now means mathematically non-monotone (neither non-decreasing nor
non-increasing over the sampled points); a non-decreasing curve is monotone (correction of S-1, decisions_round5.md).
Every number of report_round5_CN.md that is not in the Part A/B logs is on one line of logs/partC_headline.log.

Usage (from this directory): python3 headline.py   -> logs/partC_headline.log, results/headline.json
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'results'); LOGS = os.path.join(HERE, 'logs')
B = 316.8921168493721; B2 = 2*B
EPS_P1 = (0.1, 0.2, 0.5, 1.0, 2.0, 4.0); EPS_P2 = (0.1, 0.2, 0.5, 1.0, 2.0); EPS_P4 = (0.2, 0.5, 1.0, 2.0)
TABLES = (('U', 'S'), ('U', 'W'), ('P5514', 'W'))


def crossing_report(eps, vals, pred):
    """eps[0] = 0. Returns the first sampled eps where pred holds (with its neighbours), every crossing (a change of pred between
    consecutive sampled points), the direction of the sampled curve ('上升' non-decreasing, '下降' non-increasing, '常数' both,
    None neither) and the 非单调 flag in the mathematical sense: the sampled values are neither non-decreasing nor
    non-increasing (round 6 Part 0a; the round-5 rule S-1 flagged every curve that was not non-increasing, so a rising curve
    was called 非单调). More than one crossing of a threshold predicate implies 非单调."""
    f = [bool(pred(v)) for v in vals]
    cross = [(eps[i-1], eps[i]) for i in range(1, len(eps)) if f[i] != f[i-1]]
    first = next((i for i in range(len(eps)) if f[i]), None)
    up, down = nondecreasing(vals), nonincreasing(vals)
    direction = '常数' if (up and down) else ('上升' if up else ('下降' if down else None))
    nonmono = direction is None
    assert nonmono or len(cross) <= 1
    return dict(flags=f, crossings=cross, first=(eps[first] if first is not None else None),
                neighbours=((eps[first-1], eps[first]) if first not in (None, 0) else None), nonmono=nonmono, direction=direction)

def nonincreasing(v): return all(v[i] <= v[i-1] for i in range(1, len(v)))
def nondecreasing(v): return all(v[i] >= v[i-1] for i in range(1, len(v)))

def verdict(n_ok, n): return '对' if n_ok == n else ('部分对' if n_ok > 0 else '错')


def main(out):
    A = [json.loads(l) for l in open(os.path.join(RES, 'archive_partA.jsonl'))]
    ok = [r for r in A if r['status'] == 'ok']
    chk = json.load(open(os.path.join(RES, 'partA_check.json')))['results']
    PB = json.load(open(os.path.join(RES, 'partB.json'))); SA = json.load(open(os.path.join(RES, 'schemeA.json')))
    SD = json.load(open(os.path.join(RES, 'schemeD.json'))); SW = json.load(open(os.path.join(RES, 'partA_sweep.json')))
    H = {}
    out(f"C-ARCH archive lines {len(A)}, ok {len(ok)}, assertion stops {len(A)-len(ok)}")
    def get(name, tb, cls, **kw):
        return [r for r in ok if r['input'] == name and r['table'] == tb and r['cls'] == cls and all(r[k] == v for k, v in kw.items())]

    # --- g_med and eps_50 (S-1) ---
    for name, tb in TABLES:
        g0 = chk[f'{name}/{tb}']['dmD']; meds = [g0]; rngs = []
        for e in EPS_P4:
            v = [r['dmD'] for r in get(name, tb, 'P4', eps_target=e)]
            meds.append(float(np.median(v))); rngs.append((min(v), max(v), len(v)))
        rep = crossing_report((0.0,)+EPS_P4, meds, lambda g: g < g0/2)
        H[f'gmed:{name}/{tb}'] = dict(g0=g0, half=g0/2, gmed=meds, report=rep)
        for e, m, (lo, hi, n) in zip(EPS_P4, meds[1:], rngs):
            out(f"C-GMED {name}/{tb} eps={e} g_med {m!r} (min {lo!r}, max {hi!r}, n = {n}); ratio to nominal {m/g0!r}")
        tag = 'headline' if (name, tb) == ('U', 'W') else 'supplementary'
        shape_ = '非单调' if rep['nonmono'] else f"单调{rep['direction']}"
        if rep['first'] is None:
            txt = f"{shape_}，未观察到减半跨越（测试至 ε=2）"
        else:
            txt = f"{shape_}; eps_50 = {rep['first']} (neighbouring sampled points {rep['neighbours'][0]} and {rep['neighbours'][1]})"
        if rep['nonmono']: txt += f"; crossings {rep['crossings']}"
        out(f"C-EPS50 {name}/{tb} ({tag}) nominal {g0!r}, half {g0/2!r}; {txt}")

    # --- eps_B on the worst P1 sign (S-2) ---
    for name, tb in TABLES+(('U', 'SoD'), ('P5514', 'SoD')):
        e0 = chk[f'{name}/{tb}']['EN']; per = {}
        for sg in (1, -1):
            vals = [e0]+[get(name, tb, 'P1', eps_target=e, sign=sg)[0]['EN'] for e in EPS_P1]
            rep = crossing_report((0.0,)+EPS_P1, vals, lambda v: v > B)
            per[sg] = dict(EN=vals, report=rep)
            out(f"C-EPSB-SIGN {name}/{tb} sign {'+' if sg > 0 else '-'}: E[N] at eps 0, {', '.join(str(e) for e in EPS_P1)} = "
                f"{[round(v, 4) for v in vals]}; first eps with E[N] > B {rep['first']}; crossings {rep['crossings']}; 非单调 {rep['nonmono']}")
        f = {sg: per[sg]['report']['first'] for sg in (1, -1)}
        if f[1] is not None and (f[-1] is None or f[1] < f[-1]): worst = 1
        elif f[-1] is not None and (f[1] is None or f[-1] < f[1]): worst = -1
        else:
            worst = None
            for i in range(len(EPS_P1), 0, -1):
                a, b = per[1]['EN'][i], per[-1]['EN'][i]
                if a != b: worst = 1 if a > b else -1; break
        rep = per[worst]['report']
        if rep['first'] is None: txt = '测试至 ε=4 未观察到跨越'
        else: txt = f"eps_B = {rep['first']} (neighbouring sampled points {rep['neighbours'][0]} and {rep['neighbours'][1]})"
        if rep['nonmono']: txt += f"; 非单调: crossings {rep['crossings']}"
        tag = 'headline' if (name, tb) == ('U', 'W') else ('control' if tb == 'SoD' else 'supplementary')
        H[f'epsB:{name}/{tb}'] = dict(worst_sign=worst, per_sign={str(k): v for k, v in per.items()}, text=txt)
        out(f"C-EPSB {name}/{tb} ({tag}) worst P1 sign {'+' if worst > 0 else '-'}; {txt} ({'离散方案下'})")

    for name in ('U', 'P5514'):
        e0 = chk[f'{name}/SoD']['EN']; q = get(name, 'SoD', 'P1')
        out(f"C-SOD-P1 {name}/SoD over the 12 P1 points: max |E[N] - E[N](theta_0)| {max(abs(r['EN']-e0) for r in q)!r}; "
            f"E[N](theta_0) - B {e0-B!r}; points with E[N] > B {sum(r['EN'] > B for r in q)}/12 (verdicts decided at this level)")

    # --- class summaries for the report (one line each) ---
    for name, tb in TABLES:
        for cls, eps in (('P1', EPS_P1), ('P2', EPS_P2)):
            for e in eps:
                for sg in (1, -1):
                    r = get(name, tb, cls, eps_target=e, sign=sg)[0]
                    out(f"C-ROW {name}/{tb} {cls} {'+' if sg > 0 else '-'}{e} eps_att {r['eps_attained']!r} E[N] {r['EN']!r} dmD {r['dmD']!r} "
                        f"retention {r['retention']!r} E[N]<=B {r['budget_ok']} SoD E[N] {r['EN_SoD']!r}")
        p3 = get(name, tb, 'P3')
        out(f"C-ROW {name}/{tb} P3 (18 points, eps 0.5): dmD median {float(np.median([r['dmD'] for r in p3]))!r}, min {min(r['dmD'] for r in p3)!r} "
            f"({min(p3, key=lambda r: r['dmD'])['dir']} {'+' if min(p3, key=lambda r: r['dmD'])['sign'] > 0 else '-'}), max {max(r['dmD'] for r in p3)!r}; "
            f"E[N] min {min(r['EN'] for r in p3)!r}, max {max(r['EN'] for r in p3)!r}; within B {sum(r['budget_ok'] for r in p3)}/18")
        for e in EPS_P4:
            q = get(name, tb, 'P4', eps_target=e)
            out(f"C-ROW {name}/{tb} P4 eps={e}: dmD median {float(np.median([r['dmD'] for r in q]))!r}, min {min(r['dmD'] for r in q)!r}, "
                f"max {max(r['dmD'] for r in q)!r}; E[N] median {float(np.median([r['EN'] for r in q]))!r}, min {min(r['EN'] for r in q)!r}, "
                f"max {max(r['EN'] for r in q)!r}; within B {sum(r['budget_ok'] for r in q)}/{len(q)}")
    for name in ('U', 'P5514'):
        q = [r for r in ok if r['input'] == name and r['table'] == 'SoD']
        out(f"C-ROW {name}/SoD over the 80 points: E[N] min {min(r['EN'] for r in q)!r}, max {max(r['EN'] for r in q)!r}; within B "
            f"{sum(r['budget_ok'] for r in q)}/{len(q)}; log det change vs theta_0 min {min(r['dlogdet_vs_nominal'] for r in q)!r}, "
            f"max {max(r['dlogdet_vs_nominal'] for r in q)!r}")

    # --- R1 (S-3) ---
    g = H['gmed:U/W']; c1_ = g['gmed'][2] < g['half']; c2_ = g['gmed'][4] <= 0
    R = {}
    R['R1'] = dict(parts=[c1_, c2_], verdict=verdict(c1_+c2_, 2))
    out(f"C-R1 g_med(0.5) = {g['gmed'][2]!r} < half nominal {g['half']!r}: {c1_}; g_med(2) = {g['gmed'][4]!r} <= 0: {c2_} -> {R['R1']['verdict']}")
    # --- R2 (S-4) ---
    hold = 0
    for name, tb in TABLES:
        found = []
        for e in (0.1, 0.2, 0.5, 1.0):
            p, m = get(name, tb, 'P1', eps_target=e, sign=1)[0]['EN'], get(name, tb, 'P1', eps_target=e, sign=-1)[0]['EN']
            if (p > B and m < 200) or (m > B and p < 200): found.append(e)
            out(f"C-R2 {name}/{tb} eps={e}: E[N](+) {p!r}, E[N](-) {m!r}; one sign > B and the other < 200: {e in found}")
        hold += bool(found)
        out(f"C-R2 {name}/{tb} holds: {bool(found)} (at eps {found})")
    R['R2'] = dict(tables_holding=hold, verdict=verdict(hold, 3)); out(f"C-R2 tables holding {hold}/3 -> {R['R2']['verdict']}")
    # --- R3 (S-5) ---
    r3 = {}
    for name, tb in TABLES:
        g0 = chk[f'{name}/{tb}']['dmD']; q = get(name, tb, 'P4', eps_target=0.2)
        good = [r['dir'] for r in q if r['dmD'] >= 0.7*g0 and r['EN'] <= B]
        keep = sum(r['dmD'] >= 0.7*g0 for r in q); bud = sum(r['EN'] <= B for r in q)
        r3[f'{name}/{tb}'] = len(good) >= 8
        out(f"C-R3 {name}/{tb} P4 eps=0.2: directions with dmD >= 0.7 nominal ({0.7*g0!r}) {keep}/10, with E[N] <= B {bud}/10, both {len(good)}/10 "
            f"({good}); holds (>= 8): {len(good) >= 8}")
    v3 = '对' if all(r3.values()) else ('部分对' if r3['U/W'] else '错')
    R['R3'] = dict(per_table=r3, verdict=v3); out(f"C-R3 -> {v3} (U/W decides; all three for 对)")
    # --- R4 (S-6) ---
    t0 = PB['theta0']; p4 = [d for d in PB['p4'] if d['eps'] == 0.2]
    i1 = t0['B']['logdet']-t0['A']['logdet']; i2 = t0['D']['logdet']-t0['B']['logdet']
    i3 = float(np.median([d['B_vs_Ap'] for d in p4])); i4 = sum(d['B_sum_ok'] for d in p4)
    parts = [i1 >= 3, i2 <= 1, i3 >= 3, i4 >= 8]
    R['R4'] = dict(values=[i1, i2, i3, i4], parts=parts, verdict=verdict(sum(parts), 4))
    out(f"C-R4 (i) log det B - log det A = {i1!r} >= 3: {parts[0]}; (ii) log det D - log det B = {i2!r} <= 1: {parts[1]}; "
        f"(iii) P4 eps=0.2 median B vs A' = {i3!r} >= 3: {parts[2]}; (iv) directions with total <= 2B {i4}/{len(p4)} >= 8: {parts[3]} -> {R['R4']['verdict']}")
    out(f"C-R4 check (not scored): B - A' at theta_0 = {PB['check']['B_minus_Ap']!r}")
    # --- R5 (S-7) ---
    comp = []
    def ret(name, cls, **kw): return [r['retention'] for r in get(name, 'W', cls, **kw)]
    for cls, eps in (('P1', EPS_P1), ('P2', EPS_P2)):
        for e in eps:
            for sg in (1, -1):
                u, p = ret('U', cls, eps_target=e, sign=sg)[0], ret('P5514', cls, eps_target=e, sign=sg)[0]
                comp.append((f'{cls} {"+" if sg > 0 else "-"}{e}', u, p))
    comp.append(('P3 median', float(np.median(ret('U', 'P3'))), float(np.median(ret('P5514', 'P3')))))
    for e in EPS_P4: comp.append((f'P4 median eps={e}', float(np.median(ret('U', 'P4', eps_target=e))), float(np.median(ret('P5514', 'P4', eps_target=e)))))
    lower = 0
    for lab, u, p in comp:
        lower += p < u; out(f"C-R5 {lab}: retention U/W {u!r}, P5514/W {p!r}; P5514 lower: {p < u}")
    pw = 0; npw = 0
    for r in get('P5514', 'W', 'P1')+get('P5514', 'W', 'P2')+get('P5514', 'W', 'P3')+get('P5514', 'W', 'P4'):
        u = get('U', 'W', r['cls'], eps_target=r['eps_target'], sign=r['sign'], dir=r['dir'])[0]['retention']; npw += 1; pw += r['retention'] < u
    R['R5'] = dict(lower=lower, n=len(comp), pointwise_lower=pw, pointwise_n=npw, verdict=verdict(lower, len(comp)))
    out(f"C-R5 P5514/W retention lower in {lower}/{len(comp)} comparisons (pointwise supplement {pw}/{npw}) -> {R['R5']['verdict']}")
    # --- R6 (S-8) ---
    wA, wS, wD = SW['wall_s'], SA['wall_s'], SD['wall_s']; tot = wA+wS+wD
    R['R6'] = dict(walls=[wA, wS, wD], total_s=tot, verdict='对' if tot <= 5400 else '错')
    out(f"C-R6 Part A sweep {wA:.1f} s + scheme A {wS:.1f} s + scheme D lambda scan {wD:.1f} s = {tot:.1f} s = {tot/3600:.3f} h <= 1.5 h -> {R['R6']['verdict']}")
    H['R'] = R
    json.dump(H, open(os.path.join(RES, 'headline.json'), 'w'), indent=1, ensure_ascii=False, default=str)


if __name__ == '__main__':
    f = open(os.path.join(LOGS, 'partC_headline.log'), 'w')
    def out(m): print(m); f.write(m+'\n'); f.flush()
    out(f"=== headline.py started {time.strftime('%Y-%m-%d %H:%M:%S')}; spec reports_CN/plan_round5_CN.md v3.2 §4; B = {B!r}")
    main(out)
    out("=== headline.py done")
