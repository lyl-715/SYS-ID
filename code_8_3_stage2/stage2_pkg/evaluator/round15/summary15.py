"""Round 15 summary (prompt §4-§7): reads results/jobs.jsonl, setup.json, selection.json, check1.json, check2.json; writes results/round15.json
(one record per job plus the summaries) and results/fig_round15.png. No Monte Carlo is run here."""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'results'); EV = os.path.dirname(HERE)
B = 316.8921168493721; I_SOD = 197.20097668224076
LEVELS = (1, 2, 4); EPS = (0.5, 1, 2, 4)
OUTSIDE4 = {'V0@1.5768': (316.89, 18.5, 207.505), 'VF(1000,0)@1.5768': (317.07, 5.75, 207.498), 'VF(300,0)@1.5768': (316.94, 3.14, 207.476),
            'VF(100,0)@1.5768': (317.16, 1.83, 207.421)}


def jl(name): return json.load(open(os.path.join(RES, name)))

def load_jobs():
    done = {}
    for l in open(os.path.join(RES, 'jobs.jsonl')):
        if l.strip(): r = json.loads(l); done[r['key']] = r
    return done

def pct(a, q): return float(np.percentile(np.asarray(a, float), q))

def stats_set(recs):
    EN = np.array([r['EN'] for r in recs]); dI = np.array([r['dI'] for r in recs]); sd = np.array([r['sd_N'] for r in recs])
    fo = np.array([r['frac_over'] for r in recs]); d = EN-B
    out = dict(n=len(recs), frac_EN_over_B=float(np.mean(EN > B)), EN_minus_B=dict(p5=pct(d, 5), median=pct(d, 50), p95=pct(d, 95), max=float(d.max()), min=float(d.min())),
               half_width=(pct(d, 95)-pct(d, 5))/2, dI=dict(min=float(dI.min()), p5=pct(dI, 5), median=pct(dI, 50), p95=pct(dI, 95), max=float(dI.max())),
               median_sd_N=pct(sd, 50), pooled_frac_N_over_B=float(fo.mean()))
    if 'e_mean' in recs[0]:
        out.update(median_e_mean=pct([r['e_mean'] for r in recs], 50), median_e_sd=pct([r['e_sd'] for r in recs], 50),
                   median_frac_clip_hi=pct([r['frac_clip_hi'] for r in recs], 50), max_frac_clip_hi=float(max(r['frac_clip_hi'] for r in recs)),
                   max_frac_clip_lo=float(max(r['frac_clip_lo'] for r in recs)))
    return out

def true_line(r):
    out = dict(EN=r['EN'], EN_se=r['EN_se'], EN_minus_B=r['EN']-B, sd_N=r['sd_N'], frac_N_over_B=r['frac_over'], S=r['S'], S_se=r['S_se'], logdet=r['logdet'],
               logdet_batch_se=r.get('logdet_batch_se'), dI=r['dI'], R=r['R'])
    for k in ('e_mean', 'e_sd', 'frac_clip_lo', 'frac_clip_hi'):
        if k in r: out[k] = r[k]
    return out


def stage(log=print):
    done = load_jobs(); setup = jl('setup.json'); sel = jl('selection.json'); c1 = jl('check1.json'); c2 = jl('check2.json')
    V = list(setup['variants']); M = setup['models']; e0 = setup['e0']
    VMs, VFs = sel['selection']['VM']['variant'], sel['selection']['VF']['variant']; trio = ['V0', VMs, VFs]
    out = dict(e0=e0, B=B, I_SoD=I_SOD, selection=sel)
    # model summary
    out['models'] = {st: dict(n=sum(1 for m in M.values() if m['set'] == st), rms_median=pct([m['rms'] for m in M.values() if m['set'] == st], 50),
                              max_median=pct([m['max'] for m in M.values() if m['set'] == st], 50), max_max=float(max(m['max'] for m in M.values() if m['set'] == st)))
                     for st in ('main', 'ref', 'scaled', 'gross')}
    out['models_main_by_level'] = {str(c): dict(rms_median=pct([M[f'main_c{c}_d{d:02d}']['rms'] for d in range(40)], 50),
                                                max_median=pct([M[f'main_c{c}_d{d:02d}']['max'] for d in range(40)], 50),
                                                max_p95=pct([M[f'main_c{c}_d{d:02d}']['max'] for d in range(40)], 95)) for c in LEVELS}
    # 4a level tables
    lvl = {}
    for c in LEVELS:
        lvl[str(c)] = {vn: stats_set([done[f'main_c{c}_d{d:02d}|{vn}'] for d in range(40)]) for vn in V}
    out['levels'] = lvl
    out['true'] = {vn: true_line(done[f'true|{vn}']) for vn in V}
    tV0 = out['true']['V0']['dI']
    # 4c head-to-head
    out['head_to_head'] = dict(variants=trio, rows={vn: dict(levels={str(c): lvl[str(c)][vn] for c in LEVELS}, true=out['true'][vn]) for vn in trio},
                               VFstar_minus_VMstar_median_dI={str(c): lvl[str(c)][VFs]['dI']['median']-lvl[str(c)][VMs]['dI']['median'] for c in LEVELS},
                               true_VFstar_minus_VMstar_dI=out['true'][VFs]['dI']-out['true'][VMs]['dI'])
    # 4d scaled and gross
    sc = {}
    for vn in trio:
        sc[vn] = {}
        for eps in EPS:
            recs = [done[f'scaled_e{eps:g}_d{d:02d}|{vn}'] for d in range(40)]
            EN = np.array([r['EN'] for r in recs]); dI = np.array([r['dI'] for r in recs])
            sc[vn][f'{eps:g}'] = dict(frac_EN_over_B=float(np.mean(EN > B)), median_EN_minus_B=pct(EN-B, 50), max_EN_minus_B=float((EN-B).max()), median_dI=pct(dI, 50),
                                      p5_dI=pct(dI, 5), median_sd_N=pct([r['sd_N'] for r in recs], 50),
                                      **({'median_frac_clip_hi': pct([r['frac_clip_hi'] for r in recs], 50)} if 'frac_clip_hi' in recs[0] else {}))
    gross = {g: {vn: dict(EN=done[f'{g}|{vn}']['EN'], EN_over_2B=bool(done[f'{g}|{vn}']['EN'] > 2*B), dI=done[f'{g}|{vn}']['dI'], sd_N=done[f'{g}|{vn}']['sd_N'],
                          frac_clip_hi=done[f'{g}|{vn}'].get('frac_clip_hi'), frac_clip_lo=done[f'{g}|{vn}'].get('frac_clip_lo')) for vn in trio}
             for g in ('gross_plus', 'gross_minus')}
    def eps_ok(vn):
        last = None
        for eps in EPS:
            if sc[vn][f'{eps:g}']['frac_EN_over_B'] <= 0.05: last = eps
            else: break
        return last
    out['scaled'] = sc; out['gross'] = gross; out['eps_max_compliant'] = {vn: eps_ok(vn) for vn in trio}
    # reference set
    out['reference_set'] = {mn: {vn: dict(EN=done[f'{mn}|{vn}']['EN'], dI=done[f'{mn}|{vn}']['dI'], sd_N=done[f'{mn}|{vn}']['sd_N'])
                                 for vn in ('V0', 'VF(1000,0)', 'VF(300,0)', 'VF(100,0)')} for mn in M if M[mn]['set'] == 'ref'}
    # checks
    r14 = json.load(open(os.path.join(EV, 'round14', 'results', 'part1.json')))['d']['mc_up']
    t = done['true|V0']
    z3 = dict(EN=(t['EN']-r14['EN_mc'])/np.hypot(t['EN_se'], r14['EN_se']), S=(t['S']-r14['S_mc'])/np.hypot(t['S_se'], r14['S_se']),
              logdet=(t['logdet']-r14['logdet_mc'])/np.hypot(t['logdet_batch_se'], r14['logdet_batch_se']))
    c4 = {}
    for vn, (en, sd, ld) in OUTSIDE4.items():
        r = done[f'true|{vn}']
        c4[vn] = dict(EN=r['EN'], EN_se=r['EN_se'], sd_N=r['sd_N'], logdet=r['logdet'], logdet_batch_se=r['logdet_batch_se'], outside=dict(EN=en, sd_N=sd, logdet=ld),
                      diff=dict(EN=r['EN']-en, sd_N=r['sd_N']-sd, logdet=r['logdet']-ld))
    bad5 = [k for k, r in done.items() if not r['check5']]
    out['checks'] = {'1_outside_deterministic': dict(pass_=c1['pass_'], max_EN_rel=max(r['EN_rel'] for r in c1['rows'].values()),
                                                     max_logdet_rel=max(r['logdet_rel'] for r in c1['rows'].values())),
                     '2_evaluator_vs_mc': dict(pass_=c2['pass_'], z={k: v['z'] for k, v in c2['rows'].items()}, clipped={k: v['mc']['n_clipped'] for k, v in c2['rows'].items()}),
                     '3_true_model_vs_round14': dict(pass_=bool(max(abs(v) for v in z3.values()) <= 3), z=z3, round14=dict(EN=r14['EN_mc'], S=r14['S_mc'], logdet=r14['logdet_mc']),
                                                     here=dict(EN=t['EN'], S=t['S'], logdet=t['logdet'])),
                     '4_outside_mc': dict(rows=c4, note='not a gate; differences reported'),
                     '5_every_job': dict(pass_=not bad5, n_jobs=len(done), failures=bad5)}
    # questions
    Q = {}
    Q['Q1'] = {str(c): dict(frac_over=lvl[str(c)]['V0']['frac_EN_over_B'], EN_minus_B=lvl[str(c)]['V0']['EN_minus_B'], half_width=lvl[str(c)]['V0']['half_width']) for c in LEVELS}
    Q['Q2'] = {str(c): dict(median_dI=lvl[str(c)]['V0']['dI']['median'], true_dI=tV0, diff=lvl[str(c)]['V0']['dI']['median']-tV0,
                            within_0_5=bool(abs(lvl[str(c)]['V0']['dI']['median']-tV0) <= 0.5)) for c in LEVELS}
    tab = sel['table']
    Q['Q3'] = dict(complying=[vn for vn in V if vn.startswith('VM(') and tab[vn]['complies']], VMstar=VMs, complies=sel['selection']['VM']['complying'],
                   cost_c2=tV0-lvl['2'][VMs]['dI']['median'], cost_true=tV0-out['true'][VMs]['dI'])
    Q['Q4'] = dict(complying=[vn for vn in V if vn.startswith('VF(') and tab[vn]['complies']], VFstar=VFs, complies=sel['selection']['VF']['complying'],
                   cost_c2=tV0-lvl['2'][VFs]['dI']['median'], cost_true=tV0-out['true'][VFs]['dI'],
                   true_cost_VF_T0={str(T): tV0-out['true'][f'VF({T},0)']['dI'] for T in (50, 100, 300, 1000)},
                   cost_c2_all_complying={vn: tV0-lvl['2'][vn]['dI']['median'] for vn in V if vn.startswith('VF(') and tab[vn]['complies']})
    Q['Q5'] = out['head_to_head']['VFstar_minus_VMstar_median_dI']
    Q['Q6'] = {vn: {str(c): lvl[str(c)][vn]['median_sd_N'] for c in LEVELS} | {'true': out['true'][vn]['sd_N']} for vn in V}
    Q['Q7'] = dict(eps_max_compliant=out['eps_max_compliant'], scaled={vn: {e: sc[vn][e]['frac_EN_over_B'] for e in sc[vn]} for vn in trio}, gross=gross)
    out['questions'] = Q
    # predictions (scoring conventions of predictions_round15.md)
    P = []
    P.append(dict(line='check 1 reproduces to 1e-9; check 2 passes', value=dict(check1=c1['pass_'], check2=c2['pass_']), hit=bool(c1['pass_'] and c2['pass_'])))
    hw_rng = {1: (20, 50), 2: (30, 70), 4: (40, 100)}
    h = all(0.30 <= lvl[str(c)]['V0']['frac_EN_over_B'] <= 0.65 and hw_rng[c][0] <= lvl[str(c)]['V0']['half_width'] <= hw_rng[c][1] for c in LEVELS)
    P.append(dict(line='Q1: V0 fraction over budget in [30 %, 65 %] at every level; half-width of 5-95 % range in [20,50] / [30,70] / [40,100]',
                  value={str(c): dict(frac=lvl[str(c)]['V0']['frac_EN_over_B'], half_width=lvl[str(c)]['V0']['half_width']) for c in LEVELS}, hit=bool(h)))
    P.append(dict(line='Q2: yes at c = 1 and c = 2', value={c: Q['Q2'][c]['diff'] for c in Q['Q2']}, hit=bool(Q['Q2']['1']['within_0_5'] and Q['Q2']['2']['within_0_5'])))
    h = bool(sel['selection']['VM']['complying'] and not tab['VM(0.05)']['complies'] and Q['Q3']['cost_c2'] >= 0.8)
    P.append(dict(line='Q3: complying margin Delta >= 0.10; its median dI at c = 2 at least 0.8 nat below the true-model V0',
                  value=dict(complying=Q['Q3']['complying'], VMstar=VMs, cost_c2=Q['Q3']['cost_c2']), hit=h))
    h = bool(tab['VF(100,3)']['complies'] and any(v <= 0.4 for v in Q['Q4']['cost_c2_all_complying'].values())
             and all(Q['Q4']['true_cost_VF_T0'][str(T)] <= 0.15 for T in (100, 300, 1000)))
    P.append(dict(line='Q4: VF(100, 3) complies; some complying VF within 0.4 nat at c = 2; true-model cost of VF(T, 0) <= 0.15 for T >= 100',
                  value=dict(VF100_3_complies=tab['VF(100,3)']['complies'], cost_c2_complying=Q['Q4']['cost_c2_all_complying'], true_cost=Q['Q4']['true_cost_VF_T0']), hit=h))
    P.append(dict(line='Q5: VF* above VM* by at least 0.3 nat at c = 1 and c = 2', value=Q['Q5'], hit=bool(Q['Q5']['1'] >= 0.3 and Q['Q5']['2'] >= 0.3)))
    fb_small = [vn for vn in V if vn.startswith('VF(50,') or vn.startswith('VF(100,')]
    h = bool(all(lvl[str(c)][vn]['median_sd_N'] < 4 for vn in fb_small for c in LEVELS) and all(16 <= lvl[str(c)]['V0']['median_sd_N'] <= 21 for c in LEVELS))
    P.append(dict(line='Q6: median sd of N below 4 for T <= 100 and between 16 and 21 for V0',
                  value=dict(VF_T_le_100={vn: {str(c): lvl[str(c)][vn]['median_sd_N'] for c in LEVELS} for vn in fb_small}, V0={str(c): lvl[str(c)]['V0']['median_sd_N'] for c in LEVELS}), hit=h))
    h = bool(sc[VFs]['0.5']['frac_EN_over_B'] <= 0.05 and all(gross[g][vn]['EN_over_2B'] for g in gross for vn in trio))
    P.append(dict(line='Q7: VF* <= 5 % over budget up to eps = 0.5; at the 2 % gross error E[N] > 2B for every variant',
                  value=dict(VFstar_eps05=sc[VFs]['0.5']['frac_EN_over_B'], gross={g: {vn: gross[g][vn]['EN'] for vn in trio} for g in gross}), hit=h))
    out['predictions'] = dict(lines=P, hits=sum(p['hit'] for p in P), n=len(P))
    # figure
    out['figure'] = figure(lvl, trio, tV0, os.path.join(RES, 'fig_round15.png'))
    # job records
    out['jobs'] = done
    with open(os.path.join(RES, 'round15.json'), 'w') as f: json.dump(out, f, indent=1)
    # log
    log(f"MODELS main by level: {out['models_main_by_level']}; gross {[(g, M[g]['max']) for g in ('gross_plus', 'gross_minus')]}")
    for c in LEVELS:
        for vn in V:
            x = lvl[str(c)][vn]; tr = out['true'][vn]
            log(f"L c={c} {vn:11s}: frac E[N]>B {x['frac_EN_over_B']:.3f}; E[N]-B p5 {x['EN_minus_B']['p5']:+7.2f} med {x['EN_minus_B']['median']:+7.2f} p95 {x['EN_minus_B']['p95']:+7.2f} "
                f"max {x['EN_minus_B']['max']:+7.2f}; dI min {x['dI']['min']:.3f} p5 {x['dI']['p5']:.3f} med {x['dI']['median']:.3f} p95 {x['dI']['p95']:.3f}; med sd N {x['median_sd_N']:.2f}; "
                f"pooled N>B {x['pooled_frac_N_over_B']:.3f} | true: E[N]-B {tr['EN_minus_B']:+.2f} dI {tr['dI']:.3f} sd N {tr['sd_N']:.2f} N>B {tr['frac_N_over_B']:.3f}")
    log(f"SELECT VM* = {VMs}, VF* = {VFs}; VF* - VM* median dI {out['head_to_head']['VFstar_minus_VMstar_median_dI']}")
    for vn in trio:
        for eps in EPS:
            x = sc[vn][f'{eps:g}']
            log(f"SCALED {vn:11s} eps={eps:g}: frac E[N]>B {x['frac_EN_over_B']:.3f}; E[N]-B median {x['median_EN_minus_B']:+.2f} max {x['max_EN_minus_B']:+.2f}; median dI {x['median_dI']:.3f}")
        for g in gross: log(f"GROSS {g} {vn}: E[N] {gross[g][vn]['EN']:.2f} (> 2B: {gross[g][vn]['EN_over_2B']}); dI {gross[g][vn]['dI']:.3f}; clip hi {gross[g][vn]['frac_clip_hi']}")
    for k, v in out['checks'].items(): log(f"CHECK {k}: {json.dumps(v)[:500]}")
    for k, v in Q.items(): log(f"{k}: {json.dumps(v)[:700]}")
    for p in P: log(f"PREDICTION {'HIT ' if p['hit'] else 'MISS'} {p['line']}: {json.dumps(p['value'])[:500]}")
    log(f"PREDICTIONS {out['predictions']['hits']}/{out['predictions']['n']} hits")
    return out


CAPTION = ("Number of transmissions above the budget (top) and information gain over constant SoD (bottom) of the one-parameter rule when the "
           "sensor's model is an estimate with covariance c J_SoD^-1 (c = 1: one constant-SoD experiment), input U, B = 316.89: median over 40 models "
           "(point) and 5-95 % range (whisker) for no correction (V0), the selected margin (VM*) and the selected count feedback (VF*). What to look at: "
           "how far each variant's whisker reaches above 0 in the top panel, and how far its points sit below the dashed true-model line in the bottom panel.")

def figure(lvl, trio, tV0, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'; COL = ['#2a78d6', '#eb6834', '#1baf7a']
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                         'axes.facecolor': SURF, 'figure.facecolor': SURF, 'font.family': 'DejaVu Sans'})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.4, 4.4), sharex=True, constrained_layout=True)
    xs = np.arange(len(LEVELS)); off = (-0.18, 0.0, 0.18)
    for vn, col, o in zip(trio, COL, off):
        for ax, kind in ((a1, 'EN'), (a2, 'dI')):
            med, lo, hi = [], [], []
            for c in LEVELS:
                x = lvl[str(c)][vn]
                if kind == 'EN': med.append(x['EN_minus_B']['median']); lo.append(x['EN_minus_B']['p5']); hi.append(x['EN_minus_B']['p95'])
                else: med.append(x['dI']['median']); lo.append(x['dI']['p5']); hi.append(x['dI']['p95'])
            med, lo, hi = map(np.array, (med, lo, hi))
            ax.errorbar(xs+o, med, yerr=[med-lo, hi-med], fmt='o', color=col, ms=4.5, lw=1.4, capsize=2.5, mec=SURF, mew=0.8, label=vn if kind == 'EN' else None, zorder=3)
    a1.axhline(0, ls='--', color=INK2, lw=0.9, zorder=1); a2.axhline(tV0, ls='--', color=INK2, lw=0.9, zorder=1)
    a2.text(0.02, 0.03, 'dashed line: true model, V0', transform=a2.transAxes, ha='left', va='bottom', fontsize=7, color=INK2)
    a1.text(0.02, 0.98, 'dashed line: budget B', transform=a1.transAxes, ha='left', va='top', fontsize=7, color=INK2)
    a1.set_ylabel(r'$E[N] - B$ (transmissions)'); a2.set_ylabel(r'$\Delta I$ over constant SoD (nat)')
    a2.set_xticks(xs); a2.set_xticklabels([f'c = {c}' for c in LEVELS]); a2.set_xlabel(r'model error level ($\delta \sim N(0, c\,J_{SoD}^{-1})$)')
    for ax in (a1, a2):
        ax.grid(True, axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    a1.set_xlim(-0.5, len(LEVELS)-0.5)
    fig.legend(loc='outside lower center', ncol=3, frameon=False, fontsize=8, handletextpad=0.3, columnspacing=1.0)
    fig.savefig(path, dpi=300); plt.close(fig)
    return dict(path=os.path.relpath(path, HERE), caption=CAPTION)
