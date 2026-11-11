"""Round 17 summary (prompt §4-§6): reads results/partA.json and results/partB.json, writes results/round17.json; Q1-Q4, predictions (scoring
conventions of predictions_round17.md), checks. No evaluation is run here."""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'results')
DESIGNED = ('round4', 'T1', 'Tprime', 'rule')
M_LIST = (6, 12, 18, 24, 36, 48)


def stage(log=print):
    A = json.load(open(os.path.join(RES, 'partA.json'))); Bj = json.load(open(os.path.join(RES, 'partB.json')))
    D = A['designs']; order = A['order']; full = D['full']['diag']; sod = D['sod']['diag']
    out = dict(partA=A, partB=Bj)
    # Part A tables
    t1 = {}
    for k in order:
        d = D[k]; g = d['diag']
        t1[k] = dict(label=d['label'], I=g['I'], dI=d['dI'], I_minus_full=g['I_minus_full'], EN=d['EN'], S=d['S'], mu_min=g['mu_min'], mu_max=g['mu_max'],
                     mu_ratio=g['mu_ratio'], geo_mu=g['geo_mu'], cond=g['cond'], cond_rel_full=g['cond']/full['cond']-1, lam_min=g['lam_min'], lam_max=g['lam_max'],
                     sd_min=g['sd_min'], sd_geo=g['sd_geo'], sd_max=g['sd_max'], tr_ratio=g['tr_ratio'], lam_min_ratio=g['lam_min_ratio'], mu=g['mu'], sd_ratio=g['sd_ratio'])
    out['table_A'] = t1
    # Part B table
    tb = {}
    for M in M_LIST:
        r = Bj['rows'][str(M)]; g = r['diag24']; mc = r['mc']
        tb[str(M)] = dict(M_min=M, m=M/24, n=r['fit']['n'], j=r['fit']['j'], e_eff=r['fit']['e_eff'], EN24=r['rec24']['EN'], I24=r['rec24']['logdet'], S24=r['rec24']['S'],
                          EN120=r['rec120']['EN'], I120=r['rec120']['logdet'], S120=r['rec120']['S'], mu_min=g['mu_min'], mu_max=g['mu_max'], cond=g['cond'],
                          sd_min=g['sd_min'], sd_geo=g['sd_geo'], sd_max=g['sd_max'], mc_e_fit=mc['e_fit'], mc_logdet=mc['logdet_at_B'], mc_bracket=mc['bracket'],
                          outside_e_fit=r['outside']['e_fit'], outside_logdet=r['outside']['logdet'],
                          mc_minus_outside=dict(e_fit=mc['e_fit']-r['outside']['e_fit'], logdet=mc['logdet_at_B']-r['outside']['logdet']),
                          budget24=r['fit']['budget24'], budget120=r['fit']['budget120'], j_plus_1_infeasible=r['fit']['j_plus_1_infeasible'],
                          mass_dev24=r['rec24']['mass_dev_max'], mass_dev120=r['rec120']['mass_dev_max'], min_m=min(r['rec24']['min_m'], r['rec120']['min_m']))
    out['table_B'] = tb
    series = {'sigma24': {M: tb[str(M)]['I24'] for M in M_LIST}, 'sigma120': {M: tb[str(M)]['I120'] for M in M_LIST}, 'mc': {M: tb[str(M)]['mc_logdet'] for M in M_LIST}}
    mono, d1, d2 = {}, {}, {}
    for kk, sr in series.items():
        seq = [sr[M] for M in sorted(M_LIST, reverse=True)]            # 48 -> 6
        mono[kk] = bool(all(b > a for a, b in zip(seq, seq[1:])))
        d1[kk] = sr[6]-sr[24]; d2[kk] = sr[24]-sr[48]
    out['partB_summary'] = dict(monotone=mono, I6_minus_I24=d1, I24_minus_I48=d2,
                                steps={kk: {f"{a}->{b}": sr[b]-sr[a] for a, b in zip(sorted(M_LIST, reverse=True), sorted(M_LIST, reverse=True)[1:])} for kk, sr in series.items()})
    # questions
    Q = {}
    Q['Q1'] = {k: dict(mu_min=t1[k]['mu_min'], mu_max=t1[k]['mu_max'], cond=t1[k]['cond'], cond_over_full=t1[k]['cond']/full['cond']) for k in order}
    Q['Q2'] = {k: dict(mu_min_ratio=t1[k]['mu_min']/sod['mu_min'], geo_ratio=float(np.exp((t1[k]['I']-sod['I'])/9)),
                       worst_over_average=(t1[k]['mu_min']/sod['mu_min'])/float(np.exp((t1[k]['I']-sod['I'])/9))) for k in order if k not in ('full', 'sod')}
    for v in Q['Q2'].values(): v['larger_in_worst_direction'] = bool(v['worst_over_average'] > 1)
    Q['Q3'] = {k: dict(cond=t1[k]['cond'], rel_to_full=t1[k]['cond_rel_full'], within_5pct=bool(abs(t1[k]['cond_rel_full']) <= 0.05)) for k in order if k != 'full'}
    Q['Q4'] = dict(monotone=mono, I6_minus_I24=d1, I24_minus_I48=d2)
    out['questions'] = Q
    # checks
    rep = all(D[k]['reproduces'] for k in order)
    refv = all(v['pass_'] for v in A['reference_values'].values())
    c3r = Bj['rows']['24'].get('check_round14', {})
    c3 = dict(M24_bitwise=bool(c3r.get('pass_')), all_budget=all(tb[str(M)]['budget24'] and tb[str(M)]['budget120'] for M in M_LIST),
              all_j_plus_1_infeasible=all(tb[str(M)]['j_plus_1_infeasible'] for M in M_LIST),
              max_mass_dev24=max(tb[str(M)]['mass_dev24'] for M in M_LIST), max_mass_dev120=max(tb[str(M)]['mass_dev120'] for M in M_LIST),
              min_m=min(tb[str(M)]['min_m'] for M in M_LIST))
    c3['pass_'] = bool(c3['M24_bitwise'] and c3['all_budget'] and c3['all_j_plus_1_infeasible'])
    out['checks'] = {'1_reproductions': dict(pass_=bool(rep and refv), designs={k: D[k]['I_rel'] for k in order}, reference_values=A['reference_values']),
                     '2_geo_identity': dict(pass_=A['geo_identity_pass'], max_rel=max(D[k]['diag']['geo_check_rel'] for k in order)),
                     '3_partB': c3, '4_round10': A['check4']}
    # predictions
    P = []
    P.append(dict(line='All reproductions pass', value=dict(reproductions=rep, reference_values=refv, geo=A['geo_identity_pass'], M24=c3['M24_bitwise'], round10=A['check4']['pass_']),
                  hit=bool(rep and refv and A['geo_identity_pass'] and c3['M24_bitwise'] and A['check4']['pass_'])))
    ru, tp, fx, cw = t1['rule'], t1['Tprime'], t1['fixed'], t1['classW']
    parts = dict(rule=bool(0.255 <= ru['mu_min'] <= 0.270 and 0.283 <= ru['mu_max'] <= 0.297 and 97 <= ru['cond'] <= 101),
                 Tprime=bool(tp['mu_min'] >= 0.26 and tp['mu_max'] <= 0.31 and 97 <= tp['cond'] <= 101),
                 fixed=bool(all(abs(v-0.327) <= 0.001 for v in fx['mu'])), classW=bool(all(0.30 <= v <= 0.37 for v in cw['mu'])))
    P.append(dict(line='Q1/Q3: rule mu in [0.255, 0.270] / [0.283, 0.297], cond in [97, 101]; T\' mu_min >= 0.26, mu_max <= 0.31, cond in [97, 101]; '
                       'fixed threshold all mu within 0.001 of 0.327; class W all mu in [0.30, 0.37]',
                  value=dict(parts=parts, rule=[ru['mu_min'], ru['mu_max'], ru['cond']], Tprime=[tp['mu_min'], tp['mu_max'], tp['cond']], fixed=fx['mu'], classW=cw['mu']),
                  hit=bool(all(parts.values()))))
    q2 = Q['Q2']['rule']
    P.append(dict(line='Q2: mu_min(rule)/mu_min(SoD) >= 5, geometric-mean ratio about 3.1', value=q2,
                  hit=bool(q2['mu_min_ratio'] >= 5 and 3.05 <= q2['geo_ratio'] < 3.15)))
    P.append(dict(line='standard-deviation ratios: rule all in [1.80, 2.00]; constant SoD all in [2.4, 4.6]', value=dict(rule=ru['sd_ratio'], sod=t1['sod']['sd_ratio']),
                  hit=bool(all(1.80 <= v <= 2.00 for v in ru['sd_ratio']) and all(2.4 <= v <= 4.6 for v in t1['sod']['sd_ratio']))))
    h = all(mono[kk] and 0.05 <= d1[kk] <= 0.15 and 0.15 <= d2[kk] <= 0.25 for kk in series)
    P.append(dict(line='Q4: monotone; I(0.25) - I(1) in [0.05, 0.15]; I(1) - I(2) in [0.15, 0.25] (both grids and the Monte Carlo)', value=Q['Q4'], hit=bool(h)))
    out['predictions'] = dict(lines=P, hits=sum(p['hit'] for p in P), n=len(P))
    json.dump(out, open(os.path.join(RES, 'round17.json'), 'w'), indent=1)
    # log
    for k in order:
        r = t1[k]
        log(f"A {k:7s}: dI {r['dI']:+8.4f} I-I_full {r['I_minus_full']:+8.4f} E[N] {r['EN']:8.2f} S {r['S']:8.2f} mu {r['mu_min']:.4f}..{r['mu_max']:.4f} (ratio {r['mu_ratio']:.3f}) "
            f"cond {r['cond']:.3f} ({r['cond_rel_full']:+.2%}) sd {r['sd_min']:.3f}/{r['sd_geo']:.3f}/{r['sd_max']:.3f} tr {r['tr_ratio']:.3f} lmin {r['lam_min_ratio']:.3f}")
    for M in M_LIST:
        r = tb[str(M)]
        log(f"B M_min={M:2d} ({r['m']:.2f} sigma): R({r['n']}, {r['j']}) e_eff {r['e_eff']:.4f}; s24 EN {r['EN24']:.3f} I {r['I24']:.4f}; s120 EN {r['EN120']:.3f} I {r['I120']:.4f}; "
            f"mu {r['mu_min']:.4f}..{r['mu_max']:.4f} cond {r['cond']:.3f} sd {r['sd_min']:.3f}..{r['sd_max']:.3f}; MC e_fit {r['mc_e_fit']:.4f} I {r['mc_logdet']:.4f} "
            f"(outside {r['outside_e_fit']} / {r['outside_logdet']})")
    log(f"B monotone {mono}; I(6)-I(24) {d1}; I(24)-I(48) {d2}")
    for k, v in Q.items(): log(f"{k}: {json.dumps(v)[:800]}")
    for k, v in out['checks'].items(): log(f"CHECK {k}: {json.dumps(v)[:500]}")
    for p in P: log(f"PREDICTION {'HIT ' if p['hit'] else 'MISS'} {p['line']}: {json.dumps(p['value'])[:600]}")
    log(f"PREDICTIONS {out['predictions']['hits']}/{out['predictions']['n']} hits")
    return out
