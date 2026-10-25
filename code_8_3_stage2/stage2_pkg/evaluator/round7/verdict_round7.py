"""Round 7: verdict (spec reports_CN/plan_round7_CN.md v3.1 §2.4, mutually exclusive, in order) and predictions G1-G6 scored by
DECISIONs S-1 to S-6 (decisions_round7.md, written before Parts B-D ran). 固定设计、离散方案下.

Reads results/{part0_verify,partA_checks,partB_diagnose,partC_regrade,partD_match,tables_round7}.json and the run logs (wall
clock); writes logs/verdict.log and results/verdict.json.

Usage (from this directory): python3 verdict_round7.py
"""
import os, sys, json, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'results'); LOGS = os.path.join(HERE, 'logs')
B = 316.8921168493721
LOG = open(os.path.join(LOGS, 'verdict.log'), 'w')
def out(m): print(m); LOG.write(m+'\n'); LOG.flush()
def J(n): return json.load(open(os.path.join(RES, n)))

def score(subs):
    """subs: list of (name, True / False / None (未定)); 对 iff all True, 错 iff none True, else 部分对."""
    t = sum(1 for _, v in subs if v is True)
    return '对' if t == len(subs) else ('错' if t == 0 else '部分对')

def wall_from_logs():
    """sum of every logged '=== <script> done in X s' line of this round's computation logs (re-runs included) plus the Part 0
    verification total."""
    tot = {}
    p0 = open(os.path.join(LOGS, 'part0_verify.log')).read()
    m = re.search(r'total ([0-9.]+) s', p0); tot['part0_verify'] = float(m.group(1))
    for f in sorted(os.listdir(LOGS)):
        if f in ('part0_verify.log', 'verdict.log') or not f.endswith('.log'): continue
        for l in open(os.path.join(LOGS, f)):
            m = re.match(r'=== (\S+) (?:\S+ )?done in ([0-9.]+) s', l)
            if m: tot[f] = tot.get(f, 0.0)+float(m.group(2))
    return tot

def main():
    P0 = J('part0_verify.json'); A = J('partA_checks.json'); Bd = J('partB_diagnose.json'); C = J('partC_regrade.json'); D = J('partD_match.json')
    T = J('tables_round7.json'); sel = C['selection']; g = C['gate']; gr = C['grid_rows']
    out(f"V0 checks: Part 0 verification {'PASS' if P0['pass_'] else 'FAIL'}; A-CELL {'PASS' if A['cell']['pass_'] else 'FAIL'}; "
        f"A2 {'PASS' if A['a2']['pass_'] else 'FAIL'}; A3 {'PASS' if A['a3_pass'] else 'FAIL'}; A4 {'PASS' if A['a4']['pass_'] else 'FAIL'}; "
        f"T4 restoration {'PASS' if T['T4_restore']['pass_'] else 'FAIL'}")
    out(f"V0 diagnosis (B2, diagnostic): (i) {Bd['values']['i_sigma48']!r}, (ii) {Bd['values']['ii_sigma48_shift']!r}, (iii) {Bd['values']['iii_sigma72']!r}; "
        f"(i)-(iii) {Bd['d_i_iii']!r}, (ii)-(iii) {Bd['d_ii_iii']!r} -> {Bd['reading']}")
    out(f"V0 gate: max|d2| {g['max_d2']!r} at {g['at']} (<= 0.05: {g['pass_']}); max|d1| {g['max_d1']!r} at {g['d1_at']}")
    # --- verdict -----------------------------------------------------------------------------------------------------------
    items = {}; case = None; selected = sel.get('selected'); pid = None
    if not g['pass_']:
        case = 1; label = '网格未收敛'
    else:
        passers = sel['passers']; minc = sel.get('min_c')
        if not passers or minc is None or minc > 2.0:
            case = 2; label = '受测候选未找到低代价方案'
        else:
            pid = sel['policy_id']; r = gr[f'{selected}|5']
            items['a_rho26'] = (r['rho26'] >= 0.8, f"rho_26 = {r['rho26']:.4f} (>= 0.8)")
            und = [m['label'] for m in sel['main_points'] if m['undetermined']]
            items['b_main_points'] = (not und, f"main check points 精度未定: {und}")
            d3 = D.get('D3', {}) if D.get('applicable') else {}
            allm = d3.get('status') == 'ok' and d3.get('n_matched') == 20
            items['c_matched20'] = (allm, f"P4 matched {d3.get('n_matched')}/20")
            med = d3.get('median_dmD_eq') if allm else None
            items['d_median_eq'] = ((med >= 5.0) if med is not None else None, f"median Delta m_D^eq {med!r} (>= 5; 未定 unless all 20 matched)")
            if all(v is True for v, _ in items.values()): case = 3; label = '支持进一步做序贯验证'
            else: case = 4; label = '未定'
    out(f"V1 verdict (spec §2.4): case {case} -- {label}" + (f"; selected policy {selected} = {pid}" if case in (3, 4) else ''))
    for k, (v, txt) in items.items(): out(f"V1 item {k}: {'met' if v is True else ('未定' if v is None else 'not met')} -- {txt}")
    # --- predictions ---------------------------------------------------------------------------------------------------------
    G = {}
    G['G1'] = [('|(i)-(iii)| > 1', abs(Bd['d_i_iii']) > 1.0), ('|(ii)-(iii)| < 0.1', abs(Bd['d_ii_iii']) < 0.1)]
    G['G2'] = [('max|d1| <= 0.3', g['max_d1'] <= 0.3), ('max|d2| <= 0.05', g['max_d2'] <= 0.05),
               ('median d1/d2 in [8, 20]', (g['median_ratio'] is not None and 8 <= g['median_ratio'] <= 20)),
               ('sign consistent at all 135 pairs', g['sign_consistent'] == g['n_pairs'])]
    N = C['N']; L = C['L']; Pp = 'P1|theta1|+|0.5'
    if g['pass_']:
        ex = N['m010'][Pp]['5']-B
        g3i = bool(ex > 0 and 6.5 <= ex < 7.5); g3ii = bool(gr['T4|5']['budget_pass26'])
    else:
        ex = N['m010'][Pp]['5']-B; g3i = None; g3ii = None
    G['G3'] = [(f'm010 P1 +0.5 over B by ~7 on the reference grid (excess {ex:+.4f})', g3i),
               (f"T4 holds all 26 on the reference grid ({gr['T4|5']['n26_within']}/26 on sigma/120)", g3ii),
               (f"max |log det_120 - log det_72| <= 0.01 ({C['logdet_change_max_all']:.5f})", C['logdet_change_max_all'] <= 0.01)]
    dec = D.get('decomposition', {}) if D.get('applicable') else {}
    t2 = dec.get('m020', {}).get('term2_adv_change') if dec.get('m020', {}).get('status') == 'ok' else None
    cT4 = gr['T4|5']['c'] if g['pass_'] else None
    G['G4'] = [(f"|second term (m020)| <= 0.1 ({t2!r})", (abs(t2) <= 0.1) if t2 is not None else None),
               (f"T4 c on the reference grid in [1.6, 1.9] ({cT4!r})", (1.6 <= cT4 <= 1.9) if cT4 is not None else None)]
    G['G5'] = [('verdict case 3', case == 3), (f"selection rule picks T4 (picked {selected})", (selected == 'T4') if g['pass_'] else None)]
    walls = wall_from_logs(); total = sum(walls.values())
    med120 = C['wall_median']['5']
    n_resc = D.get('D3', {}).get('n_rescan', 0) if D.get('applicable') else 0
    allow = 6*3600+n_resc*841*med120/4
    pf = D.get('per_forward_sum_s', {}) if D.get('applicable') else {}
    curve_s = next((v for k, v in pf.items() if k.startswith('D1')), None); local_s = next((v for k, v in pf.items() if k == 'D3 local windows'), None)
    counts_ok = (D.get('applicable') and D.get('new', {}).get('D1 nominal curve M = 120..960 at theta_0') == 841
                 and D.get('new', {}).get('D3 local windows') in (820, None)) if D.get('applicable') else None
    G['G6'] = [(f"counts (T4 27 at n = 24; 162 + 162; curve 841; local 20 x 41 = 820)", counts_ok),
               (f"sigma/120 forward ~10 s (median {med120:.1f} s in [5, 20])", 5 <= med120 <= 20),
               (f"curve and local stages ~2.3 h serial (per-forward sums {curve_s!r}, {local_s!r} s in [1.15, 4.6] h)",
                (all(1.15*3600 <= v <= 4.6*3600 for v in (curve_s, local_s)) if (curve_s is not None and local_s is not None) else None)),
               (f"total wall <= 6 h + rescan allowance ({total:.0f} s vs {allow:.0f} s)", total <= allow)]
    res = dict(case=case, label=label, selected=selected, policy_id=pid, items={k: dict(met=v, text=t) for k, (v, t) in items.items()},
               predictions={}, wall_s=walls, wall_total_s=total)
    for k, subs in G.items():
        sc = score(subs); res['predictions'][k] = dict(score=sc, subs=[dict(claim=a, value=b) for a, b in subs])
        out(f"V2 {k}: {sc} -- " + '; '.join(f"{a}: {'对' if b is True else ('未定' if b is None else '错')}" for a, b in subs))
    for k, v in walls.items(): out(f"V3 wall {k} {v:.1f} s")
    out(f"V3 total logged wall clock {total:.1f} s = {total/3600:.3f} h")
    json.dump(res, open(os.path.join(RES, 'verdict.json'), 'w'), indent=1, ensure_ascii=False)
    return res


if __name__ == '__main__':
    main()
