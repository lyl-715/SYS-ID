"""Round 6: the verdict of spec §6 and predictions M1-M7 against the outcomes (spec reports_CN/plan_round6_CN.md v3.2, frozen;
scoring readings DECISION S-1 .. S-7 in decisions_round6.md, written before the Part A runs). Reads only the result files of
Parts 0, A, B, C and the logs' "done in" lines; no evaluation. 固定设计、离散方案下.

Usage (from this directory): python3 verdict_round6.py   -> logs/verdict.log, results/verdict.json
"""
import os, re, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'results'); LOGS = os.path.join(HERE, 'logs')
R5 = os.path.join(os.path.dirname(HERE), 'round5')
B = 316.8921168493721
MT = ('m005', 'm010', 'm020'); M_OF = {'m000': 0.0, 'm005': 0.05, 'm010': 0.10, 'm020': 0.20}


def verdict(n_ok, n): return '对' if n_ok == n else ('部分对' if n_ok > 0 else '错')

def main(out):
    F1 = json.load(open(os.path.join(RES, 'margin_final.json'))); SA = json.load(open(os.path.join(RES, 'partA_summary.json')))
    SW = json.load(open(os.path.join(RES, 'partA_sweep.json'))); PB = json.load(open(os.path.join(RES, 'partB_refine.json')))
    PC = json.load(open(os.path.join(RES, 'partC_equal_send.json'))); L5 = json.load(open(os.path.join(R5, 'results', 'schemeD_load.json')))
    rows = SA['rows']; sel = SA['selected']; mstar = SA['mstar']
    V = {}
    # --- checks of spec §4 ---
    out(f"V-CHK1 m = 0 table at theta_0 equals round 5 bit for bit: {SW['regression']['pass']}; 80-point regression of m000 and SoD vs round 5: "
        f"{SW['regression']['pass_80']}")
    for t in MT:
        d = F1[t]
        out(f"V-CHK2 {t} E[N](theta_0) {d['EN']!r} <= B_m {d['B_m']!r}: {d['feasible_Bm']}; B_m - E[N] {d['slack_Bm']!r} (<= 0.5 utilisation target: "
            f"{d['utilisation_target_met']}, not a pass condition)")
    out(f"V-CHK3 Part A evaluations: {SW['n_ok']}/{SW['n']} ok, assertion stops {SW['n_stop']}; Part B assertion failures "
        f"{sum(r['assert_fail'] for r in PB['rows'])}; Part C evaluations in the archive {PC['n_evals']}")
    out(f"V-CHK4 round-5 fixes: scheme D reproduced from the committed npz bit for bit: {L5['bitwise_all']}")
    # --- verdict (spec §6) ---
    cost = (rows['m000']['dmD0']-rows[mstar]['dmD0']) if mstar else None
    case1 = (mstar is None) or (cost > 2.0)
    a = rows[sel]['rho26'] >= 0.8
    b = not PB['main_undetermined']
    c = PC['n_matched_P4'] == 20
    d_ = (PC['median_dmD_eq_P4'] >= 5.0) if c else None
    out(f"V-6 case 1 test: m* = {mstar}; nominal cost Delta m_D(0; theta_0) - Delta m_D(m*; theta_0) = {cost!r} (> 2 nat: {cost > 2.0 if cost is not None else 'n/a'})")
    out(f"V-6 (a) rho_26({sel}) = {rows[sel]['rho26']!r} >= 0.8: {a}")
    out(f"V-6 (b) main grid-check points 精度未定: {PB['main_undetermined']} -> none: {b}")
    out(f"V-6 (c) P4 points matched {PC['n_matched_P4']}/20: {c}")
    out(f"V-6 (d) median Delta m_D^eq over the 20 P4 points >= 5 nat: {d_ if d_ is not None else '未定 (not all 20 matched)'}"
        + (f" (median {PC['median_dmD_eq_P4']!r})" if c else ''))
    if case1:
        case = '受测档位未找到低代价方案'; failed = []
    elif a and b and c and d_:
        case = '支持进一步做序贯验证'; failed = []
    else:
        case = '未定'
        failed = [k for k, v in (('(a)', a), ('(b)', b), ('(c)', c), ('(d)', d_)) if not v]
    out(f"V-6 VERDICT {case}" + (f"; not met: {failed}; 精度未定 points {PB['main_undetermined']}; unmatched {PC['unmatched']}" if case == '未定' else ''))
    if c:
        sup = (f"所有 20 个 P4 点匹配成功；Delta m_D - Delta m_D^eq 的中位数 {PC['median_diff_P4']:.4f} nat"
               + ('，匹配发送成本后，相对于原固定阈值 SoD 的比较优势明显缩小' if PC['median_diff_P4'] > 2 else '（不超过 2 nat，补充句不适用）'))
    else:
        ms_ = PC.get('matched_subset')
        sup = (f"在匹配成功的 {PC['n_matched_P4']}/20 个 P4 点中，Delta m_D - Delta m_D^eq 的中位数为 {ms_['median_diff']:.4f} nat" if ms_ else
               '没有匹配成功的 P4 点')
    out(f"V-6 supplementary {sup}")
    V['verdict'] = dict(case=case, failed=failed, a=a, b=b, c=c, d=d_, cost=cost, supplementary=sup)
    # --- M1 (S-1) ---
    m1 = [F1[t]['feasible_Bm'] and F1[t]['slack_Bm'] <= 0.5 for t in MT]
    V['M1'] = verdict(sum(m1), 3)
    out(f"V-M1 B_m - E[N](theta_0) <= 0.5: {dict(zip(MT, [round(F1[t]['slack_Bm'], 6) for t in MT]))} -> {dict(zip(MT, m1))} -> {V['M1']}")
    # --- M2 (S-2) ---
    over5 = rows['m005']['over26']
    loc_ok = all((o.startswith('P4 ') and o.endswith('+0.5')) or o == 'P1 theta1 +0.5' for o in over5)
    m2 = [mstar == 'm010', len(over5) >= 2 and loc_ok, rows['m010']['budget_pass26'],
          rows['m020']['budget_pass26'] and rows['m020']['n_P4eps1_P1pm1_within_B'] == '12/12']
    V['M2'] = verdict(sum(m2), 4)
    out(f"V-M2 (i) m* = 0.10: {m2[0]} (m* = {M_OF.get(mstar)}); (ii) m = 0.05 26-point crossings {len(over5)} {over5}, all at P4 eps = 0.5 or "
        f"P1 +0.5: {loc_ok} -> {m2[1]}; (iii) m = 0.10 holds 26: {m2[2]} ({rows['m010']['n26_within_B']}/26); (iv) m = 0.20 holds 26 "
        f"({rows['m020']['n26_within_B']}/26) and P4 eps = 1 + P1 +-1 ({rows['m020']['n_P4eps1_P1pm1_within_B']}, over {rows['m020']['P4eps1_P1pm1_over']}): "
        f"{m2[3]} -> {V['M2']}")
    # --- M3 (S-3) ---
    d0 = rows['m000']['dmD0']; m3 = []
    for t in MT:
        thr = d0-6*M_OF[t]; v = rows[t]['dmD0']; m3.append(v >= thr)
        out(f"V-M3 {t} Delta m_D(theta_0) {v!r} >= {thr!r} (printed {round(thr, 4)}): {v >= thr}; cost {d0-v!r} nat = {(d0-v)/M_OF[t]!r} m")
    V['M3'] = verdict(sum(m3), 3); out(f"V-M3 -> {V['M3']}")
    # --- M4 (S-4) ---
    m4 = []
    for t in ('m000',)+MT:
        r_ = rows[t]['rho26'] >= 0.8; g = rows[t]['gmed'][1:]; mono = all(g[i] >= g[i-1] for i in range(1, len(g)))
        m4 += [r_, mono]
        out(f"V-M4 {t} rho_26 {rows[t]['rho26']!r} >= 0.8: {r_}; g_med(0.2, 0.5, 1, 2) {[round(v, 4) for v in g]} non-decreasing: {mono} "
            f"(with eps = 0: {rows[t]['gmed_shape_with0']})")
    V['M4'] = verdict(sum(m4), 8); out(f"V-M4 {sum(m4)}/8 table-claims hold -> {V['M4']}")
    # --- M5 (S-5) ---
    mains = [r for r in PB['rows'] if r['main']]; m5 = [abs(r['dEN']) <= 0.5 and not r['flip'] for r in mains]
    for r, ok in zip(mains, m5):
        out(f"V-M5 {r['point']} ({'; '.join(r['roles'])}) |dE[N]| {abs(r['dEN'])!r} <= 0.5 and no flip ({r['verdict24']} / {r['verdict48']}): {ok}")
    V['M5'] = verdict(sum(m5), len(m5)); out(f"V-M5 -> {V['M5']} (table {sel})")
    # --- M6 (S-6) ---
    i1 = PC['n_matched'] == 21
    i2 = (PC['median_dmD_eq_P4'] >= 6.0) if c else None; i3 = (PC['median_diff_P4'] <= 1.5) if c else None
    parts = [i1, i2, i3]
    if c: V['M6'] = verdict(sum(bool(x) for x in parts), 3)
    else: V['M6'] = '部分对' if any(x for x in parts if x) else '错'
    out(f"V-M6 (i) all 21 matched: {i1} ({PC['n_matched']}/21); (ii) median Delta m_D^eq over P4 >= 6.0: "
        f"{i2 if i2 is not None else '未定'}{' ('+repr(PC['median_dmD_eq_P4'])+')' if c else ''}; (iii) median Delta m_D - Delta m_D^eq <= 1.5: "
        f"{i3 if i3 is not None else '未定'}{' ('+repr(PC['median_diff_P4'])+')' if c else ''} -> {V['M6']}")
    # --- M7 (S-7) ---
    walls = {}
    for f in sorted(os.listdir(LOGS)):
        for l in open(os.path.join(LOGS, f)):
            mm = re.search(r'^=== (\S+) (\S+) done in ([0-9.]+) s', l) or re.search(r'^=== (\S+) done in ([0-9.]+) s', l)
            if mm: walls[f] = walls.get(f, 0.0)+float(mm.groups()[-1])
    for f in ('partB_schemeD-load.log',):
        for l in open(os.path.join(R5, 'logs', f)):
            mm = re.search(r'done in ([0-9.]+) s', l)
            if mm: walls['round5/'+f] = float(mm.group(1))
    tot = sum(walls.values())
    for k, v in walls.items(): out(f"V-M7 wall {k} {v:.1f} s")
    V['M7'] = '对' if tot <= 2.5*3600 else '错'; V['wall_total_s'] = tot; V['walls'] = walls
    out(f"V-M7 total logged wall {tot:.1f} s = {tot/3600:.3f} h <= 2.5 h -> {V['M7']}")
    json.dump(V, open(os.path.join(RES, 'verdict.json'), 'w'), indent=1, ensure_ascii=False, default=str)


if __name__ == '__main__':
    f = open(os.path.join(LOGS, 'verdict.log'), 'w')
    def out(m): print(m); f.write(m+'\n'); f.flush()
    out(f"=== verdict_round6.py started {time.strftime('%Y-%m-%d %H:%M:%S')}; spec reports_CN/plan_round6_CN.md v3.2 §5, §6; B = {B!r}")
    main(out)
    out("=== verdict_round6.py finished (reads results only)")
