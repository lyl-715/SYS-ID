"""Round 16 summary (prompt §4, §5, §3, §7): distributions, extremes, correlations, phase-only / power-only sets, figure, answers to Q1-Q7,
prediction scoring (conventions of predictions_round16.md), checks 1-4. Reads results/{anchors,random,derived,floor}.json, writes
results/round16.json (every record with every run + everything derived) and results/fig_round16.png. Descriptive only.
"""
import os, json, time
import numpy as np
from scipy.stats import spearmanr
import random16 as r16
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'results')
B = r16.B
QTY = ('c_rule', 'c_sod', 'gain', 'e_fit', 'D_fit', 'info_per_send_rule', 'info_per_send_sod')
COL_RULE, COL_SOD, INK, INK2 = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'


def stats(v):
    v = np.asarray(v, np.float64); p5, p50, p95 = np.percentile(v, [5, 50, 95])
    return dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std(ddof=1)), min=float(v.min()), p5=float(p5), median=float(np.median(v)),
                p95=float(p95), max=float(v.max()), range_5_95=float(p95-p5), range_minmax=float(v.max()-v.min()))

def prank(v, ref):
    ref = np.asarray(ref, np.float64); return float(100.0*((ref < v).sum()+0.5*(ref == v).sum())/ref.size)

def col(recs, k): return np.array([r['derived'][k] for r in recs])
def icol(recs, k): return np.array([r['input'][k] for r in recs])

def statline(name, st, n=4):
    return (f"{name:>22s}: mean {st['mean']:.{n}f} sd {st['sd']:.{n}f} min {st['min']:.{n}f} p5 {st['p5']:.{n}f} median {st['median']:.{n}f} "
            f"p95 {st['p95']:.{n}f} max {st['max']:.{n}f} | p95-p5 {st['range_5_95']:.{n}f} max-min {st['range_minmax']:.{n}f}")


def figure(c_rule, c_sod, anc_rule, anc_sod, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'legend.fontsize': 8, 'font.family': 'DejaVu Sans'})
    xr = c_rule-np.median(c_rule); xs = c_sod-np.median(c_sod)
    ar = np.asarray(anc_rule)-np.median(c_rule); as_ = np.asarray(anc_sod)-np.median(c_sod)
    allv = np.concatenate([xr, xs, ar, as_]); w = 0.05
    lo = np.floor(allv.min()/w-0.5)*w-w/2; hi = np.ceil(allv.max()/w+0.5)*w+w/2
    edges = np.arange(np.round(lo/w-0.5)*w+w/2, hi+w, w)          # bin centres on multiples of 0.05 (0 is a bin centre)
    hr, _ = np.histogram(xr, edges); hs, _ = np.histogram(xs, edges)
    fig, ax = plt.subplots(figsize=(3.4, 2.9), dpi=300)
    ymax = max(hr.max(), hs.max())
    ax.stairs(hs, edges, fill=True, color=COL_SOD, alpha=0.30, linewidth=0)
    ax.stairs(hs, edges, color=COL_SOD, linewidth=1.0, label='constant SoD: $c_{sod}$ − median')
    ax.stairs(hr, edges, fill=True, color=COL_RULE, alpha=0.30, linewidth=0)
    ax.stairs(hr, edges, color=COL_RULE, linewidth=1.0, label='one-parameter rule: $c_{rule}$ − median')
    y1, y2 = -0.07*ymax, -0.15*ymax                                 # anchor ticks below the baseline: rule row, SoD row
    ax.plot(ar, np.full(ar.size, y1), linestyle='none', marker='|', markersize=7, markeredgewidth=1.0, color=COL_RULE)
    ax.plot(as_, np.full(as_.size, y2), linestyle='none', marker='|', markersize=7, markeredgewidth=1.0, color=COL_SOD)
    ax.plot([], [], linestyle='none', marker='|', markersize=7, color=INK2, label='six anchors (ticks below the axis)')
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_ylim(-0.21*ymax, 1.08*ymax)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(r'$c - \mathrm{median}(c)$ over the 200 random inputs (nat)')
    ax.set_ylabel('number of inputs (of 200)')
    ax.tick_params(colors=INK2, length=3, width=0.6)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(INK2); ax.spines[sp].set_linewidth(0.6)
    ax.set_yticks([t for t in ax.get_yticks() if t >= 0 and t <= ymax*1.08])
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), frameon=False, handlelength=1.4, borderaxespad=0.2, labelcolor=INK)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return dict(path=os.path.relpath(path, HERE), bins_nat=w, n_bins=int(len(edges)-1), edges=[float(edges[0]), float(edges[-1])],
                max_count_rule=int(hr.max()), max_count_sod=int(hs.max()), width_in=3.4, font_pt=8)


def main(out):
    t0 = time.perf_counter()
    A = json.load(open(os.path.join(RES, 'anchors.json'))); Rd = json.load(open(os.path.join(RES, 'random.json')))
    Dv = json.load(open(os.path.join(RES, 'derived.json'))); Fl = json.load(open(os.path.join(RES, 'floor.json')))
    anc = [A['records'][n] for n in r16.ANCHORS]
    rnd = [Rd['records'][f"r{i:03d}"] for i in range(r16.N_RANDOM)]
    ph = [Dv['records'][f"ph{i:03d}"] for i in range(r16.N_DERIVED)]; pw = [Dv['records'][f"pw{i:03d}"] for i in range(r16.N_DERIVED)]
    flo = [Fl['records'][f"U@{sd}"] for sd in r16.SEEDS_FLOOR]
    assert len(rnd) == 200 and len(ph) == 50 and len(pw) == 50 and len(flo) == 10
    # consistency of the derived sets with the random set
    for i in range(r16.N_DERIVED):
        assert ph[i]['input']['chi'] == rnd[i]['input']['chi'] and ph[i]['input']['p'] == [0.25]*4
        assert pw[i]['input']['p'] == rnd[i]['input']['p'] and pw[i]['input']['chi'] == [0.0]*4
    U = A['records']['U']['derived']
    res = dict(prompt='reports_CN/prompt_round16_random_inputs_EN.md', B=B, R=r16.R_MC, seed_noise=r16.SEED_NOISE, seed_inputs=r16.SEED_INPUTS,
               draws=Rd['draws'], rejected=Rd['rejected'], accepted=Rd['accepted'], versions=Rd.get('versions'))
    out(f"SUMMARY random set: {Rd['draws']} draws, {Rd['rejected']} rejected (min p < {r16.P_MIN}), {Rd['accepted']} accepted")

    # --- 4a distributions -------------------------------------------------------------------------------------------------
    dist = {k: stats(col(rnd, k)) for k in QTY}
    dist['c_rule_minus_U'] = stats(col(rnd, 'c_rule')-U['c_rule']); dist['c_sod_minus_U'] = stats(col(rnd, 'c_sod')-U['c_sod'])
    res['distribution'] = dist
    out("4a distribution over the 200 random inputs (nat; e_fit, D_fit in sigma):")
    for k, st in dist.items(): out('  '+statline(k, st))
    # --- 4b extremes ------------------------------------------------------------------------------------------------------
    ext = {}
    for k in ('c_rule', 'c_sod'):
        v = col(rnd, k); o = np.argsort(v, kind='stable')
        def row(i):
            d = rnd[i]['input']; return dict(name=d['name'], value=float(v[i]), other=float(col(rnd, 'c_sod' if k == 'c_rule' else 'c_rule')[i]),
                                             p=d['p'], chi=d['chi'], range_sigma=d['range_sigma'], tv_sigma=d['tv_sigma'], e_fit=rnd[i]['derived']['e_fit'],
                                             D_fit=rnd[i]['derived']['D_fit'])
        ext[k] = dict(lowest=[row(i) for i in o[:5]], highest=[row(i) for i in o[::-1][:5]])
        for side in ('lowest', 'highest'):
            for r_ in ext[k][side]:
                out(f"4b {k} {side:>7s}: {r_['name']} {k} {r_['value']:+.4f} p {r16_fmt(r_['p'], 3)} chi {r16_fmt(r_['chi'], 3)} range {r_['range_sigma']:.1f} "
                    f"TV {r_['tv_sigma']:.1f}")
    res['extremes'] = ext
    # --- 4c correlations --------------------------------------------------------------------------------------------------
    feats = dict(tv_sigma=icol(rnd, 'tv_sigma'), range_sigma=icol(rnd, 'range_sigma'), max_p=icol(rnd, 'max_p'), min_p=icol(rnd, 'min_p'),
                 I_full=icol(rnd, 'I_full'))
    corr = {k: {f: float(spearmanr(col(rnd, k), fv)[0]) for f, fv in feats.items()} for k in ('c_rule', 'c_sod')}
    res['spearman'] = corr
    for k in corr: out(f"4c Spearman {k}: " + ', '.join(f"{f} {v:+.3f}" for f, v in corr[k].items()))
    extra = dict(c_rule_vs_c_sod=float(spearmanr(col(rnd, 'c_rule'), col(rnd, 'c_sod'))[0]),
                 e_fit_vs_tv=float(spearmanr(col(rnd, 'e_fit'), feats['tv_sigma'])[0]),
                 gain_vs_tv=float(spearmanr(col(rnd, 'gain'), feats['tv_sigma'])[0]),
                 D_fit_vs_range=float(spearmanr(col(rnd, 'D_fit'), feats['range_sigma'])[0]))
    res['spearman_extra'] = extra
    out("4c extra (beyond the prompt's list, descriptive): Spearman " + ', '.join(f"{k} {v:+.3f}" for k, v in extra.items()))
    res['feature_ranges'] = {f: stats(fv) for f, fv in feats.items()}
    res['max_step_sigma'] = stats(icol(rnd, 'max_step_sigma'))
    # --- 4d derived sets --------------------------------------------------------------------------------------------------
    sets = {}
    for nm, recs in (('phase_only', ph), ('power_only', pw), ('random_first50', rnd[:50])):
        sets[nm] = {k: stats(col(recs, k)) for k in ('c_rule', 'c_sod', 'e_fit', 'D_fit', 'gain')}
        for k in ('c_rule', 'c_sod'): out('4d '+statline(f"{nm} {k}", sets[nm][k]))
    sets_desc = {}
    for nm, recs in (('phase_only', ph), ('power_only', pw), ('random_first50', rnd[:50]), ('random', rnd)):
        tv = icol(recs, 'tv_sigma')
        sets_desc[nm] = dict(tv=stats(tv), rho_c_rule_tv=float(spearmanr(col(recs, 'c_rule'), tv)[0]), rho_c_sod_tv=float(spearmanr(col(recs, 'c_sod'), tv)[0]))
        out(f"4d {nm:>14s}: TV min {sets_desc[nm]['tv']['min']:.0f} p5 {sets_desc[nm]['tv']['p5']:.0f} p95 {sets_desc[nm]['tv']['p95']:.0f} max "
            f"{sets_desc[nm]['tv']['max']:.0f}; Spearman with TV: c_rule {sets_desc[nm]['rho_c_rule_tv']:+.3f}, c_sod {sets_desc[nm]['rho_c_sod_tv']:+.3f}")
    res['sets_4d'] = sets; res['sets_4d_tv'] = sets_desc
    # --- percentiles of the anchors (Q6) ----------------------------------------------------------------------------------
    cr, cs = col(rnd, 'c_rule'), col(rnd, 'c_sod')
    anc_pr = {n: dict(c_rule=A['records'][n]['derived']['c_rule'], c_sod=A['records'][n]['derived']['c_sod'],
                      prank_rule=prank(A['records'][n]['derived']['c_rule'], cr), prank_sod=prank(A['records'][n]['derived']['c_sod'], cs),
                      e_fit=A['records'][n]['derived']['e_fit'], gain=A['records'][n]['derived']['gain'],
                      prank_gain=prank(A['records'][n]['derived']['gain'], col(rnd, 'gain'))) for n in r16.ANCHORS}
    tvr, rgr = icol(rnd, 'tv_sigma'), icol(rnd, 'range_sigma')
    for n in r16.ANCHORS:
        d_ = A['records'][n]['input']
        anc_pr[n].update(tv_sigma=d_['tv_sigma'], range_sigma=d_['range_sigma'], prank_tv=prank(d_['tv_sigma'], tvr), prank_range=prank(d_['range_sigma'], rgr))
    res['anchors'] = anc_pr
    out("Q6 anchors' TV / output range and their percentile ranks within the random set: " +
        '; '.join(f"{n} TV {v['tv_sigma']:.0f} ({v['prank_tv']:.1f}) range {v['range_sigma']:.0f} ({v['prank_range']:.1f})" for n, v in anc_pr.items()))
    for n, v in anc_pr.items():
        out(f"Q6 anchor {n:>6s}: c_rule {v['c_rule']:+.4f} percentile {v['prank_rule']:.1f}; c_sod {v['c_sod']:+.4f} percentile {v['prank_sod']:.1f}; "
            f"gain {v['gain']:.3f} percentile {v['prank_gain']:.1f}")
    # --- figure -----------------------------------------------------------------------------------------------------------
    res['figure'] = figure(cr, cs, [anc_pr[n]['c_rule'] for n in r16.ANCHORS], [anc_pr[n]['c_sod'] for n in r16.ANCHORS],
                           os.path.join(RES, 'fig_round16.png'))
    out(f"FIGURE {res['figure']}")

    # --- questions --------------------------------------------------------------------------------------------------------
    dr_, ds_ = dist['c_rule'], dist['c_sod']
    worst_rule = float(np.max(np.abs(cr-dr_['median']))); worst_sod = float(np.max(np.abs(cs-ds_['median'])))
    ef = dist['e_fit']; g = col(rnd, 'gain'); n85 = int((g >= 8.5).sum())
    Q = {}
    Q['Q1'] = dict(sd=dr_['sd'], p5=dr_['p5'], p95=dr_['p95'], range_5_95=dr_['range_5_95'], worst_from_median=worst_rule,
                   worst_input=rnd[int(np.argmax(np.abs(cr-dr_['median'])))]['input']['name'], min=dr_['min'], max=dr_['max'], range_minmax=dr_['range_minmax'])
    Q['Q2'] = dict(sd=ds_['sd'], p5=ds_['p5'], p95=ds_['p95'], range_5_95=ds_['range_5_95'], worst_from_median=worst_sod,
                   worst_input=rnd[int(np.argmax(np.abs(cs-ds_['median'])))]['input']['name'], min=ds_['min'], max=ds_['max'], range_minmax=ds_['range_minmax'])
    Q['Q3'] = dict(ratio=dr_['range_5_95']/ds_['range_5_95'], answer=bool(dr_['range_5_95'] <= ds_['range_5_95']/3.0))
    Q['Q4'] = dict(p5=ef['p5'], median=ef['median'], p95=ef['p95'], up=ef['p95']-ef['median'], down=ef['median']-ef['p5'], min=ef['min'], max=ef['max'],
                   answer=bool(ef['p95']-ef['median'] <= 0.03 and ef['median']-ef['p5'] <= 0.03))
    Q['Q5'] = dict(n_ge_85=n85, frac=n85/200.0, min=float(g.min()), max=float(g.max()), median=float(np.median(g)), answer=bool(n85 >= 190),
                   min_input=rnd[int(np.argmin(g))]['input']['name'], max_input=rnd[int(np.argmax(g))]['input']['name'])
    Q['Q6'] = anc_pr
    Q['Q7'] = {nm: dict(range_sod=sets[nm]['c_sod']['range_5_95'], range_rule=sets[nm]['c_rule']['range_5_95'], sd_sod=sets[nm]['c_sod']['sd'],
                        sd_rule=sets[nm]['c_rule']['sd'], ratio_sod_rule=sets[nm]['c_sod']['range_5_95']/sets[nm]['c_rule']['range_5_95'])
               for nm in sets}
    Q['Q7']['random'] = dict(range_sod=ds_['range_5_95'], range_rule=dr_['range_5_95'], sd_sod=ds_['sd'], sd_rule=dr_['sd'],
                             ratio_sod_rule=ds_['range_5_95']/dr_['range_5_95'])
    res['questions'] = Q
    out(f"Q1 c_rule: sd {dr_['sd']:.4f} nat; 5-95 % [{dr_['p5']:+.4f}, {dr_['p95']:+.4f}] range {dr_['range_5_95']:.4f} nat; worst |c - median| "
        f"{worst_rule:.4f} nat ({Q['Q1']['worst_input']}); min-max range {dr_['range_minmax']:.4f}")
    out(f"Q2 c_sod: sd {ds_['sd']:.4f} nat; 5-95 % [{ds_['p5']:+.4f}, {ds_['p95']:+.4f}] range {ds_['range_5_95']:.4f} nat; worst |c - median| "
        f"{worst_sod:.4f} nat ({Q['Q2']['worst_input']}); min-max range {ds_['range_minmax']:.4f}")
    out(f"Q3 range ratio c_rule / c_sod (5-95 %) {Q['Q3']['ratio']:.4f} (<= 1/3: {Q['Q3']['answer']})")
    out(f"Q4 e_fit p5 {ef['p5']:.4f} median {ef['median']:.4f} p95 {ef['p95']:.4f} (+{Q['Q4']['up']:.4f} / -{Q['Q4']['down']:.4f}); min {ef['min']:.4f} "
        f"max {ef['max']:.4f} -> within +-0.03: {Q['Q4']['answer']}")
    for i in np.argsort(g)[:int((g < 8.5).sum())]:
        d_ = rnd[int(i)]['input']; out(f"Q5 gain < 8.5: {d_['name']} gain {g[i]:.4f} c_rule {cr[i]:+.4f} c_sod {cs[i]:+.4f} p {r16_fmt(d_['p'], 3)} "
                                      f"range {d_['range_sigma']:.1f} TV {d_['tv_sigma']:.1f} D_fit {rnd[int(i)]['derived']['D_fit']:.3f}")
    tv_rank = np.argsort(np.argsort(icol(rnd, 'tv_sigma'))) + 1
    out("Q5 rank (1 = smallest) of TV of the inputs with gain < 8.5: " + ', '.join(f"{rnd[int(i)]['input']['name']} {int(tv_rank[i])}" for i in np.where(g < 8.5)[0]))
    out(f"Q5 gain >= 8.5 nat for {n85}/200 inputs; min {Q['Q5']['min']:.4f} ({Q['Q5']['min_input']}) median {Q['Q5']['median']:.4f} max "
        f"{Q['Q5']['max']:.4f} ({Q['Q5']['max_input']}) -> {Q['Q5']['answer']}")
    for nm, v in Q['Q7'].items():
        out(f"Q7 {nm:>14s}: 5-95 % range c_sod {v['range_sod']:.4f} c_rule {v['range_rule']:.4f} (ratio {v['ratio_sod_rule']:.2f}); "
            f"sd c_sod {v['sd_sod']:.4f} c_rule {v['sd_rule']:.4f}")

    # --- predictions (conventions of predictions_round16.md) --------------------------------------------------------------
    c1 = A['check1']
    P = {}
    P['check1'] = dict(hit=bool(c1['pass_']), values=dict(max_diff_rule=max(abs(v['diff_rule']) for v in c1['rows'].values()),
                                                         max_diff_sod=max(abs(v['diff_sod']) for v in c1['rows'].values())))
    P['Q1'] = dict(hit=bool(dr_['range_5_95'] <= 0.30 and dr_['sd'] <= 0.10 and worst_rule <= 0.45),
                   values=dict(range_5_95=dr_['range_5_95'], sd=dr_['sd'], worst=worst_rule))
    P['Q2'] = dict(hit=bool(0.6 <= ds_['range_5_95'] <= 1.8), values=dict(range_5_95=ds_['range_5_95']))
    P['Q3'] = dict(hit=Q['Q3']['answer'], values=dict(ratio=Q['Q3']['ratio']))
    P['Q4'] = dict(hit=bool(Q['Q4']['answer'] and 1.565 <= ef['median'] <= 1.590), values=dict(up=Q['Q4']['up'], down=Q['Q4']['down'], median=ef['median']))
    P['Q5'] = dict(hit=bool(n85 >= 190 and 9.2 <= Q['Q5']['median'] <= 10.0), values=dict(n_ge_85=n85, median=Q['Q5']['median']))
    pr_rule = [anc_pr[n]['prank_rule'] for n in r16.ANCHORS]
    P['Q6'] = dict(hit=bool(anc_pr['U']['prank_sod'] <= 10.0 and all(2.0 <= v <= 98.0 for v in pr_rule)),
                   values=dict(prank_sod_U=anc_pr['U']['prank_sod'], prank_rule={n: anc_pr[n]['prank_rule'] for n in r16.ANCHORS}))
    P['Q7'] = dict(hit=bool(Q['Q7']['phase_only']['range_sod'] >= 4.0*Q['Q7']['phase_only']['range_rule']),
                   values=dict(ratio=Q['Q7']['phase_only']['ratio_sod_rule']))
    res['predictions'] = P
    nh = sum(v['hit'] for v in P.values())
    for k, v in P.items(): out(f"PRED {k}: {'hit' if v['hit'] else 'MISS'} {json.dumps(v['values'], default=r16._jdef)}")
    out(f"PRED total {nh} hit / {len(P)-nh} miss")

    # --- checks -----------------------------------------------------------------------------------------------------------
    allrecs = dict(anchors=anc, random=rnd, phase_only=ph, power_only=pw, floor=flo)
    nrun = 0; bad = []; nfit = 0; ext_rule = []; ext_sod = []; wmax = -np.inf; wmin = np.inf; cmax = -np.inf; nmin = np.inf; walls = []
    for setn, recs in allrecs.items():
        for rec in recs:
            for part in ('rule', 'sod'):
                for run in rec[part]['runs']:
                    nrun += 1; walls.append(run['wall_s'])
                    wmin = min(wmin, run['W_min']); wmax = max(wmax, run['W_max']); cmax = max(cmax, run['c_max']); nmin = min(nmin, run['N_min'])
                    if not (0.0 <= run['W_min'] and run['W_max'] <= 1.0 and run['N_min'] >= 1 and np.isfinite(run['logdet']) and run['logdet_sign'] > 0):
                        bad.append((setn, rec['input']['name'], part, run['par']))
                f = rec[part]['fit']; nfit += 1
                if not (f['EN_pair'][0] >= B >= f['EN_pair'][1]): bad.append((setn, rec['input']['name'], part, 'bracket'))
            if rec['rule']['extended']: ext_rule.append((rec['input']['name'], rec['seed'], rec['rule']['extended']))
            if rec['sod']['extended']: ext_sod.append((rec['input']['name'], rec['seed'], rec['sod']['extended']))
    gse = dict(c_rule=stats([r['derived']['c_rule_group_se'] for r in rnd]), c_sod=stats([r['derived']['c_sod_group_se'] for r in rnd]))
    res['checks'] = dict(check1=A['check1'], check2=dict(sd=Fl['sd'], mean=Fl['mean'],
                                                        values={k: [r['derived'][k] for r in flo] for k in ('c_rule', 'c_sod', 'e_fit', 'D_fit', 'gain')}),
                         check3=A['check3'],
                         check4=dict(n_runs=nrun, n_fits=nfit, failures=bad, W_min=wmin, W_max=wmax, c_max=cmax, N_min=nmin, pass_=not bad),
                         extensions=dict(rule=ext_rule, sod=ext_sod), group_se_random=gse)
    out(f"CHECK4 {nrun} runs, {nfit} interpolations: W_k in [{wmin:.3e}, {wmax!r}], max per-trajectory c {cmax!r}, min N {nmin:.0f}; "
        f"failures {bad} -> {'PASS' if not bad else 'FAIL'}")
    out(f"CHECK4 list extensions: rule {len(ext_rule)} inputs {ext_rule}; SoD {len(ext_sod)} inputs {[(n, e) for n, sd, e in ext_sod]}")
    out(f"CHECK2 noise floor (U, ten seeds, no common random numbers between seeds): sd " + ', '.join(f"{k} {v:.4f}" for k, v in Fl['sd'].items()))
    out(f"DIAG within-run group SE (10 groups of 200, random set, median): c_rule {gse['c_rule']['median']:.4f}, c_sod {gse['c_sod']['median']:.4f} nat")
    res['wall'] = dict(run_median_s=float(np.median(walls)), run_min_s=float(np.min(walls)), run_max_s=float(np.max(walls)),
                       stage_s=dict(anchors=A.get('wall_s'), random=Rd.get('wall_s'), derived=Dv.get('wall_s'), floor=Fl.get('wall_s')))
    out(f"WALL runs: median {res['wall']['run_median_s']:.2f} s (min {res['wall']['run_min_s']:.2f}, max {res['wall']['run_max_s']:.2f}); stages "
        + ', '.join(f"{k} {v:.0f} s" for k, v in res['wall']['stage_s'].items() if v is not None))
    res['records'] = dict(anchors={r['input']['name']: r for r in anc}, random={r['input']['name']: r for r in rnd},
                          phase_only={r['input']['name']: r for r in ph}, power_only={r['input']['name']: r for r in pw},
                          floor={f"U@{r['seed']}": r for r in flo})
    r16.jdump(res, os.path.join(RES, 'round16.json'))
    out(f"SUMMARY written results/round16.json ({os.path.getsize(os.path.join(RES, 'round16.json'))/1e6:.1f} MB) in {time.perf_counter()-t0:.1f} s")


def r16_fmt(v, n=3): return '(' + ', '.join(f"{x:.{n}f}" for x in v) + ')'
