"""Round 13 Part A cross-input summary (prompt reports_CN/prompt_round13_partA_input_invariance_EN.md §5-§8), verdict and prediction
scoring (round13/predictions_partA.md). Reads results/partA_<name>.json of the six inputs; writes results/partA_summary.json and
results/fig_partA.png. No evaluator is imported: every number comes from the per-input files.

All derived quantities on sigma/120 (Part A §4 item 5, DECISION 13-19); sigma/24 values are listed alongside.
"""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'results')
INPUTS = ('U', 'P5514', 'SCH', 'LO', 'HI', 'N')
TIE = 0.1                                   # rank ties: values within 0.1 nat (Part A §5)
B = 316.8921168493721


def load():
    d = {}
    for n in INPUTS:
        p = os.path.join(RES, f'partA_{n}.json')
        d[n] = json.load(open(p)) if os.path.exists(p) else None
    return d

def rel(a, b, tie=TIE):
    """pairwise relation with ties: +1 / -1 if a - b > tie / < -tie, else 0."""
    x = a-b
    return 0 if abs(x) <= tie else (1 if x > 0 else -1)

def order_string(vals, tie=TIE):
    """inputs in decreasing order; consecutive values within `tie` are joined by '=' (display only, chained)."""
    ks = sorted(vals, key=lambda k: -vals[k]); s = ks[0]
    for a, b in zip(ks, ks[1:]): s += (' = ' if vals[a]-vals[b] <= tie else ' > ')+b
    return s

def rank_compare(v1, v2, names):
    """DECISION 13-21: two orders agree iff the tie-aware pairwise relation is the same for all pairs; discordant pairs = strictly
    opposite; mixed pairs = tied in one order, strict in the other."""
    disc, mixed = [], []
    for i, a in enumerate(names):
        for b in names[i+1:]:
            r1, r2 = rel(v1[a], v1[b]), rel(v2[a], v2[b])
            if r1*r2 < 0: disc.append(f'{a}-{b}')
            elif (r1 == 0) != (r2 == 0): mixed.append(f'{a}-{b}')
    return dict(order_1=order_string(v1), order_2=order_string(v2), agree=bool(not disc and not mixed), discordant_pairs=disc,
                mixed_tie_pairs=mixed, n_pairs=len(names)*(len(names)-1)//2)


def summary(log=print):
    D = load()
    missing = [n for n in INPUTS if D[n] is None or D[n].get('status') != 'done']
    names = [n for n in INPUTS if n not in missing]
    rows = {}
    for n in names:
        r = D[n]; m = r['matched']; dz = r['design']; b = dz['best']; dv = r['derived']; sd = r['sod_delta']
        rows[n] = dict(I_full=r['I_full'], G=r['sanity']['G'], range_sigma=r['sanity']['range_sigma'], max_step_sigma=r['sanity']['max_step_sigma'],
                       sod_delta=dict(EN24=sd['EN24'], I24=sd['I24'], S24=sd['S24'], EN120=sd['EN120'], I120=sd['I120'], S120=sd['S120']),
                       matched=dict(M=m['M_star'], M_sigma=m['M_star']/24, EN120=m['EN120'], residual120=m['residual120'], S120=m['S120'], I120=m['logdet120'],
                                    EN24=m['EN24'], S24=m['S24'], I24=m['logdet24'], c_sod=dv['c_sod'], c_sod24=dv['sigma24']['c_sod'],
                                    info_per_send=dv['info_per_send_SoD']),
                       table=dict(label=b['label'], T_star=dz['T_star'], lam=b['lam'], EN120=b['EN120'], S120=b['S120'], I120=b['logdet120'],
                                  EN24=b['EN24'], S24=b['S24'], I24=b['logdet24'], c_tab=dv['c_tab'], c_tab24=dv['sigma24']['c_tab'],
                                  gap=dz['certificate']['gap'], gap_two_scans=dz['certificate_two_scans']['gap'], UBp=dz['certificate']['UBp'],
                                  third_scan=('j2' in dz['scans']), info_per_send=dv['info_per_send_table'],
                                  shape=(None if r['shape'].get('skipped') else dict(q10=r['shape']['shape']['edge_q10'], q50=r['shape']['shape']['edge_median'],
                                                                                     q90=r['shape']['shape']['edge_q90'])),
                                  shape_skipped=bool(r['shape'].get('skipped')), shape_reason=r['shape'].get('reason')),
                       gain=dv['gain'], gain24=dv['sigma24']['gain'], all_checks_pass=r['all_checks_pass'])
    # inputs stopped before a designed table but with an accepted matched SoD: their constant-SoD numbers are complete (DECISION 13-28)
    partial = {}
    for n in missing:
        r = D[n]
        if r is None or 'matched' not in r or not r['matched'].get('accepted'): continue
        m = r['matched']; sd = r['sod_delta']; lp = os.path.join(HERE, 'logs', f'partA_{n}.log')
        stops = [l.strip() for l in open(lp) if l.startswith('STOP ')] if os.path.exists(lp) else []
        partial[n] = dict(I_full=r['I_full'], G=r['sanity']['G'], range_sigma=r['sanity']['range_sigma'], max_step_sigma=r['sanity']['max_step_sigma'],
                          sod_delta=dict(EN24=sd['EN24'], I24=sd['I24'], S24=sd['S24'], EN120=sd['EN120'], I120=sd['I120'], S120=sd['S120']),
                          matched=dict(M=m['M_star'], M_sigma=m['M_star']/24, EN120=m['EN120'], residual120=m['residual120'], S120=m['S120'], I120=m['logdet120'],
                                       EN24=m['EN24'], S24=m['S24'], I24=m['logdet24'], c_sod=m['logdet120']-r['I_full'], c_sod24=m['logdet24']-r['I_full'],
                                       info_per_send=(m['S120']-1.0)/(m['EN120']-1.0)),
                          stopped=(stops[-1] if stops else 'not finished'))
    out = dict(inputs=list(INPUTS), done=names, missing=missing, partial=partial, grid='sigma/120 (sigma/24 alongside)', rows=rows)
    if missing:
        log(f"PARTA-SUMMARY missing or unfinished inputs: {missing}; constant-SoD numbers available for {list(partial)}")
    if 'U' not in rows: return out
    U = rows['U']
    sodall = dict(**{n: rows[n] for n in names}, **partial); sod_names = [n for n in INPUTS if n in sodall]
    ct = {n: rows[n]['table']['c_tab'] for n in names}; cs = {n: sodall[n]['matched']['c_sod'] for n in sod_names}
    It = {n: rows[n]['table']['I120'] for n in names}; Is = {n: sodall[n]['matched']['I120'] for n in sod_names}; If = {n: sodall[n]['I_full'] for n in sod_names}
    St = {n: rows[n]['table']['S120'] for n in names}
    out['sod_inputs'] = sod_names
    out['range_c_tab'] = max(ct.values())-min(ct.values()); out['range_c_sod'] = max(cs.values())-min(cs.values())
    out['range_c_sod_table_inputs'] = max(cs[n] for n in names)-min(cs[n] for n in names)
    out['range_c_tab24'] = max(rows[n]['table']['c_tab24'] for n in names)-min(rows[n]['table']['c_tab24'] for n in names)
    out['S_ratio'] = {n: St[n]/St['U']-1 for n in names}; out['S_ratio_max_abs'] = max(abs(v) for v in out['S_ratio'].values())
    out['S_ratio24_max_abs'] = max(abs(rows[n]['table']['S24']/U['table']['S24']-1) for n in names)
    out['rank_table'] = rank_compare({n: If[n] for n in names}, It, names); out['rank_sod'] = rank_compare(If, Is, sod_names)
    pw = {}
    for lab, a, b_, kind in (('SCH - U', 'SCH', 'U', 'I'), ('P5514 - U', 'P5514', 'U', 'I'), ('c(LO) - c(HI)', 'LO', 'HI', 'c')):
        if a in rows and b_ in rows:
            if kind == 'I': pw[lab] = dict(table=It[a]-It[b_], sod=Is[a]-Is[b_], full=If[a]-If[b_])
            else: pw[lab] = dict(table=ct[a]-ct[b_], sod=cs[a]-cs[b_])
        elif a in sodall and b_ in sodall:
            if kind == 'I': pw[lab] = dict(table=None, sod=Is[a]-Is[b_], full=If[a]-If[b_])
            else: pw[lab] = dict(table=None, sod=cs[a]-cs[b_])
    out['pairwise'] = pw
    out['gain'] = {n: rows[n]['gain'] for n in names}
    out['info_per_send'] = {n: dict(table=(rows[n]['table']['info_per_send'] if n in rows else None), sod=sodall[n]['matched']['info_per_send']) for n in sod_names}
    # verdict (prompt §6)
    complete = not missing
    H1 = out['S_ratio_max_abs'] <= 0.03
    H2 = out['range_c_tab'] <= 0.5
    H3 = bool(all(k in pw for k in ('SCH - U', 'P5514 - U', 'c(LO) - c(HI)')) and abs(pw['SCH - U']['table']) <= 0.2 and abs(pw['P5514 - U']['table']) <= 0.2
              and abs(pw['c(LO) - c(HI)']['table']) <= 0.5)
    gaps = {n: rows[n]['table']['gap'] for n in names}
    unresolved = [n for n in names if gaps[n] > 0.5]
    if not complete:
        verdict = (f"incomplete: no designed table for {', '.join(missing)}; on the {len(names)} inputs with a table "
                   f"H1 {'holds' if H1 else 'fails'}, H2 {'holds' if H2 else 'fails'}, H3 {'holds' if H3 else 'fails'}")
    elif H1 and H2 and H3: verdict = 'separation supported'
    elif out['range_c_tab'] > 1.5: verdict = 'not supported'
    else:
        failed = [h for h, ok in (('H1', H1), ('H2', H2), ('H3', H3)) if not ok]
        verdict = 'partial (failed: ' + ', '.join(failed) + ')'
    if unresolved: verdict += f'; unresolved for input {", ".join(unresolved)} (certificate gap > 0.5 nat)'
    out['verdict'] = dict(H1=dict(pass_=bool(H1), value=out['S_ratio_max_abs'], rule='max_i |S_table(i)/S_table(U) - 1| <= 0.03 (sigma/120)'),
                          H2=dict(pass_=bool(H2), value=out['range_c_tab'], rule='range c_tab <= 0.5 nat (sigma/120)'),
                          H3=dict(pass_=bool(H3), values={k: v['table'] for k, v in pw.items()},
                                  rule='|I_table(SCH) - I_table(U)| <= 0.2, |I_table(P5514) - I_table(U)| <= 0.2, |c_tab(LO) - c_tab(HI)| <= 0.5 (sigma/120)'),
                          resolution=dict(pass_=not unresolved, gaps=gaps, unresolved=unresolved, rule='certificate gap <= 0.5 nat for every table'),
                          verdict=verdict)
    # check 6
    if 'LO' in rows and 'HI' in rows:
        out['check6_LO_HI_I_full'] = dict(diff=If['LO']-If['HI'], pass_=bool(abs(If['LO']-If['HI']) <= 0.01))
    out['checks_all_inputs_pass'] = all(rows[n]['all_checks_pass'] for n in names)
    out['predictions'] = score(rows, out, names, missing)
    return out


def score(rows, out, names, missing=()):
    """predictions_partA.md, scoring conventions 1-7 (fixed before running). A line that needs an input without a designed table is
    'undetermined' unless the available inputs already make it a miss (DECISION 13-28)."""
    It24 = {n: rows[n]['table']['I24'] for n in names}
    iv = {'P5514': (207.05, 207.65), 'SCH': (207.05, 207.65), 'LO': (206.08, 206.68), 'HI': (206.08, 206.68), 'N': (206.46, 207.06)}
    pw = out['pairwise']; P = []
    P.append(dict(line='H1 holds (S_table within +-3 % of S_table(U), sigma/120)', value=out['S_ratio_max_abs'], hit=bool(out['S_ratio_max_abs'] <= 0.03)))
    pts = {n: dict(I24=It24[n], interval=iv[n], in_=bool(iv[n][0] <= It24[n] <= iv[n][1])) for n in iv if n in It24}
    P.append(dict(line='H2 holds (range c_tab <= 0.5) and sigma/24 I_table points 207.35 (P5514, SCH), 206.38 (LO, HI), 206.76 (N), each +-0.3',
                  value=dict(range_c_tab=out['range_c_tab'], points=pts), hit=bool(out['range_c_tab'] <= 0.5 and all(v['in_'] for v in pts.values()))))
    P.append(dict(line='H3 holds', value={k: v['table'] for k, v in pw.items()}, hit=bool(out['verdict']['H3']['pass_'])))
    d = pw.get('c(LO) - c(HI)', {}).get('sod')
    P.append(dict(line='constant SoD not separable: range c_sod >= 0.7 and |c_sod(LO) - c_sod(HI)| >= 0.5', value=dict(range_c_sod=out['range_c_sod'], LO_HI=d),
                  hit=bool(out['range_c_sod'] >= 0.7 and d is not None and abs(d) >= 0.5)))
    v = pw.get('P5514 - U', {}).get('sod')
    P.append(dict(line='matched SoD: I_SoD(P5514) - I_SoD(U) in [0.5, 0.9]', value=v, hit=bool(v is not None and 0.5 <= v <= 0.9)))
    g2 = {n: rows[n]['table']['gap_two_scans'] for n in names}
    P.append(dict(line='certificate gap <= 0.3 nat for every input after two scans', value=g2, hit=bool(all(x <= 0.3 for x in g2.values()))))
    g = out['gain']
    P.append(dict(line='gain = I_table - I_SoD in [9, 11.5] for every input (sigma/120)', value=g, hit=bool(all(9 <= x <= 11.5 for x in g.values()))))
    needs_all = {0, 1, 5, 6}                                   # lines 1, 2, 6, 7 need every input's table; 3-5 do not
    for i, p in enumerate(P):
        p['result'] = 'hit' if p['hit'] else 'miss'
        if missing and i in needs_all and p['hit']:
            p['result'] = 'undetermined'; p['hit'] = None; p['note'] = f"no designed table for {', '.join(missing)}; holds on the other inputs"
    return dict(lines=P, hits=sum(p['result'] == 'hit' for p in P), misses=sum(p['result'] == 'miss' for p in P),
                undetermined=sum(p['result'] == 'undetermined' for p in P), n=len(P))


CAPTION = ("Information of the designed table (filled) and of the matched constant SoD (open) against the full-data information, "
           "both relative to the uniform input U, sigma/120, budget B = 316.89; matched SoD for six inputs, designed table for the inputs "
           "whose design finished (N: none, its scan stopped on a conservation assertion). What to look at: table points on the dotted line "
           "y = x mean that the designed table loses the same amount of information for every input (separation); the matched-SoD points "
           "are shown for contrast.")

def figure(out, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'; C1, C2 = '#2a78d6', '#eb6834'
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                         'axes.facecolor': SURF, 'figure.facecolor': SURF, 'font.family': 'DejaVu Sans'})
    rows = out['rows']; names = out['done']; U = rows['U']
    sodall = dict(**rows, **out.get('partial', {})); snames = out.get('sod_inputs', names)
    x = {n: sodall[n]['I_full']-U['I_full'] for n in snames}
    yt = {n: rows[n]['table']['I120']-U['table']['I120'] for n in names}
    ys = {n: sodall[n]['matched']['I120']-U['matched']['I120'] for n in snames}
    fig, ax = plt.subplots(figsize=(3.4, 3.6), constrained_layout=True)
    allv = list(x.values())+list(yt.values())+list(ys.values())
    lo, hi = min(allv), max(allv); sp_ = (hi-lo if hi > lo else 1.0)
    lim = (lo-0.30*sp_, hi+0.30*sp_)                             # room for the direct labels on both sides
    ax.plot(lim, lim, ls=':', color=INK2, lw=1.0, zorder=1, label='y = x')
    ax.scatter([x[n] for n in names], [yt[n] for n in names], s=30, color=C1, edgecolor=SURF, linewidth=1.0, zorder=3, label='designed table')
    ax.scatter([x[n] for n in snames], [ys[n] for n in snames], s=30, facecolor='none', edgecolor=C2, linewidth=1.4, zorder=3, label='matched SoD')
    # direct labels: points of one series closer than 4 % of the axis range share one label
    span = lim[1]-lim[0]
    for ys_, dxs, ha in ((yt, -6, 'right'), (ys, 6, 'left')):
        left = list(ys_); groups = []
        while left:
            a = left.pop(0); g = [a]
            for b in list(left):
                if abs(x[a]-x[b]) < 0.04*span and abs(ys_[a]-ys_[b]) < 0.04*span: g.append(b); left.remove(b)
            groups.append(g)
        for g in groups:
            gx = np.mean([x[n] for n in g]); gy = np.mean([ys_[n] for n in g])
            ax.annotate(', '.join(g), (gx, gy), xytext=(dxs, 0), textcoords='offset points', ha=ha, va='center', fontsize=8, color=INK)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel(r'$I_{\mathrm{full}} - I_{\mathrm{full}}(U)$ (nat)')
    ax.set_ylabel(r'$I - I(U)$ (nat), $\sigma/120$')
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    fig.legend(loc='outside lower center', ncol=3, frameon=False, fontsize=8, handletextpad=0.3, columnspacing=0.9)
    fig.savefig(path, dpi=300); plt.close(fig)
    return dict(path=os.path.relpath(path, HERE), caption=CAPTION, x=x, y_table=yt, y_sod=ys)


def stage(log=print):
    out = summary(log)
    if 'U' in out['rows'] and len(out['done']) >= 2:
        out['figure'] = figure(out, os.path.join(RES, 'fig_partA.png'))
    with open(os.path.join(RES, 'partA_summary.json'), 'w') as f: json.dump(out, f, indent=1)
    for n in out['done']:
        r = out['rows'][n]; t = r['table']; m = r['matched']
        log(f"ROW {n}: I_full {r['I_full']:.4f}; SoD(Delta*) EN24 {r['sod_delta']['EN24']:.2f} I24 {r['sod_delta']['I24']:.4f}; matched M* {m['M']} "
            f"EN120 {m['EN120']:.3f} I120 {m['I120']:.4f} c_sod {m['c_sod']:.4f}; table EN120 {t['EN120']:.3f} S120 {t['S120']:.2f} I120 {t['I120']:.4f} "
            f"(sigma/24 {t['I24']:.4f}) c_tab {t['c_tab']:.4f} gap {t['gap']:.4f} (two scans {t['gap_two_scans']:.4f}); gain {r['gain']:.4f}")
    for n, r in out.get('partial', {}).items():
        m = r['matched']
        log(f"ROW {n} (no designed table: {r['stopped']}): I_full {r['I_full']:.4f}; SoD(Delta*) EN24 {r['sod_delta']['EN24']:.2f} I24 {r['sod_delta']['I24']:.4f}; "
            f"matched M* {m['M']} EN120 {m['EN120']:.3f} I120 {m['I120']:.4f} c_sod {m['c_sod']:.4f}")
    if 'verdict' in out:
        log(f"RANGES c_tab {out['range_c_tab']:.4f} ({len(out['done'])} inputs); c_sod {out['range_c_sod']:.4f} ({len(out['sod_inputs'])} inputs), "
            f"{out['range_c_sod_table_inputs']:.4f} (the {len(out['done'])} inputs with a table); S ratio max {out['S_ratio_max_abs']:.4f}")
        log(f"RANK table: by I_full {out['rank_table']['order_1']} | by I_table {out['rank_table']['order_2']} | agree {out['rank_table']['agree']} "
            f"(discordant {out['rank_table']['discordant_pairs']}, mixed {out['rank_table']['mixed_tie_pairs']})")
        log(f"RANK SoD: by I_full {out['rank_sod']['order_1']} | by I_SoD {out['rank_sod']['order_2']} | agree {out['rank_sod']['agree']} "
            f"(discordant {out['rank_sod']['discordant_pairs']}, mixed {out['rank_sod']['mixed_tie_pairs']})")
        log(f"PAIRWISE {out['pairwise']}")
        log(f"VERDICT H1 {out['verdict']['H1']['pass_']} H2 {out['verdict']['H2']['pass_']} H3 {out['verdict']['H3']['pass_']} resolution "
            f"{out['verdict']['resolution']['pass_']} -> {out['verdict']['verdict']}")
        for p in out['predictions']['lines']: log(f"PREDICTION {p['result'].upper()} {p['line']}: {p['value']}")
        log(f"PREDICTIONS {out['predictions']['hits']} hit, {out['predictions']['misses']} miss, {out['predictions']['undetermined']} undetermined of {out['predictions']['n']}; check 6 {out.get('check6_LO_HI_I_full')}; "
            f"all per-input checks {out['checks_all_inputs_pass']}")
    return out


if __name__ == '__main__':
    stage()
