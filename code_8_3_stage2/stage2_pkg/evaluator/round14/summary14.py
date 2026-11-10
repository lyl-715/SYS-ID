"""Round 14 summary (prompt §4-§7): collects results/part1.json, budgets.json, input_<name>.json into results/round14.json; answers Q1-Q5,
scores predictions_round14.md (scoring conventions fixed there), lists checks 1-5. No evaluator is run here."""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'results'); EV = os.path.dirname(HERE)
B_NOM = 316.8921168493721
INPUTS = ('U', 'SCH', 'P5514', 'LO', 'HI', 'N')
E_PRED = {160.0: 1.88, 240.0: 1.70, B_NOM: 1.58, 400.0: 1.45, 480.0: 1.35}


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None

def stage(log=print):
    p1 = load('part1.json'); bu = load('budgets.json')
    ins = {n: load(f'input_{n}.json') for n in INPUTS if n != 'U'}
    out = dict(part1=p1, budgets=bu, inputs=ins)
    # --- input table (U from part 1 + round 13 Part A) ---
    paU = json.load(open(os.path.join(EV, 'round13', 'results', 'partA_U.json')))['derived']
    b = p1['b']
    rows_in = {'U': dict(n=b['fit']['n'], j=b['fit']['j'], e_eff=b['fit']['e_eff'], delta=b['fit']['delta'], I_rule=b['rec120']['logdet'], I_rule24=b['rec24']['logdet'],
                         c_rule=b['rec120']['logdet']-paU['I_full'], rule_minus_table=b['rec120']['logdet']-paU['I_table'], rule_minus_SoD=b['rec120']['logdet']-paU['I_SoD'],
                         EN120=b['rec120']['EN'], S120=b['rec120']['S'], I_full=paU['I_full'], I_table=paU['I_table'], I_SoD=paU['I_SoD'])}
    for n, r in ins.items():
        if r is None: continue
        d = r['derived']
        rows_in[n] = dict(n=r['fit']['n'], j=r['fit']['j'], e_eff=d['e_eff'], delta=d['delta'], I_rule=d['I_rule'], I_rule24=d['I_rule24'], c_rule=d['c_rule'],
                          rule_minus_table=d['rule_minus_table'], rule_minus_SoD=d['rule_minus_SoD'], EN120=d['EN120'], S120=d['S120'],
                          I_full=r['partA']['I_full'], I_table=r['partA']['I_table'], I_SoD=r['partA']['I_SoD'])
    out['input_table'] = rows_in
    missing_in = [n for n in INPUTS if n not in rows_in]
    # --- budget table ---
    rows_b = {}
    for k, row in (bu['rows'].items() if bu else []):
        if row.get('done'): rows_b[k] = dict(B=row['B'], n=row['fit']['n'], j=row['fit']['j'], M_star=row['matched']['M_star'],
                                              SoD_residual=row['matched']['residual120'], SoD_accepted=row['matched']['accepted'], **row['derived'])
    out['budget_table'] = rows_b
    missing_b = [B for B in E_PRED if f"{B:.10g}" not in rows_b]
    # --- questions ---
    c = p1['c']; d = p1['d']
    Q = {}
    Q['Q1'] = dict(answer=bool(p1['a']['pass_'] and b['pass_'] and c['pass_']), a=p1['a']['pass_'], b=b['pass_'], c=c['pass_'],
                   z=dict(EN=c['z_EN'], S=c['z_S'], logdet=c['z_logdet']), n_clipped=c['n_clipped'])
    Q['Q2'] = dict(answer=bool(d['Q2']), EN=d['mc_up']['EN_mc'], EN_se=d['mc_up']['EN_se'], EN_minus_B=d['EN_minus_B'], logdet=d['mc_up']['logdet_mc'],
                   lattice120=b['rec120']['logdet'], diff=d['logdet_minus_lattice120'], e_up=d['e_up'], e_C=d['e_C'])
    deltas = {**{f"B={k}": v['delta'] for k, v in rows_b.items()}, **{f"input {n}": v['delta'] for n, v in rows_in.items()}}
    Q['Q3'] = dict(answer=bool(all(0 < v <= 0.10 for v in deltas.values()) and not missing_b and not missing_in), deltas=deltas,
                   min=min(deltas.values()), max=max(deltas.values()), missing=dict(budgets=missing_b, inputs=missing_in))
    dIs = {k: v['dI'] for k, v in rows_b.items()}
    Q['Q4'] = dict(answer=bool(all(v >= 8 for v in dIs.values()) and not missing_b), dI=dIs, min=(min(dIs.values()) if dIs else None))
    cr = {n: v['c_rule'] for n, v in rows_in.items()}; rt = {n: v['rule_minus_table'] for n, v in rows_in.items()}
    rng = max(cr.values())-min(cr.values())
    Q['Q5'] = dict(answer=bool(rng <= 0.1 and all(abs(v) <= 0.3 for v in rt.values()) and not missing_in), range_c_rule=rng, c_rule=cr, rule_minus_table=rt)
    out['questions'] = Q
    # --- predictions ---
    P = []
    a_ok = p1['a']['pass_'] and b['pass_']
    P.append(dict(line='1a, 1b reproduce to 1e-9; fit_rule(B_nom) = (37, 13)', value=dict(a=p1['a']['pass_'], b=b['check'], n_j=(b['fit']['n'], b['fit']['j'])), hit=bool(a_ok)))
    P.append(dict(line='1c: |z| <= 3 on all three', value=Q['Q1']['z'], hit=bool(max(abs(v) for v in Q['Q1']['z'].values()) <= 3)))
    P.append(dict(line='1d: e_C in [1.56, 1.60]; continuous log det within 0.05 of 207.498', value=dict(e_C=d['e_C'], logdet=d['mc_up']['logdet_mc']),
                  hit=bool(1.56 <= d['e_C'] <= 1.60 and abs(d['mc_up']['logdet_mc']-207.498) <= 0.05)))
    if not missing_b:
        h = all(0.02 <= v['delta'] <= 0.08 and abs(v['e_eff']-E_PRED[v['B']]) <= 0.04 and 9 <= v['dI'] <= 12 and 1.2 <= v['classW_minus_rule'] <= 2.0 for v in rows_b.values())
        P.append(dict(line='2a: delta in [0.02, 0.08]; e_eff 1.88/1.70/1.58/1.45/1.35 +-0.04; dI in [9, 12]; class-W - rule in [1.2, 2.0]',
                      value={k: dict(delta=v['delta'], e_eff=v['e_eff'], e_pred=E_PRED[v['B']], dI=v['dI'], classW_minus_rule=v['classW_minus_rule']) for k, v in rows_b.items()},
                      hit=bool(h)))
    else:
        P.append(dict(line='2a', value=f'missing budgets {missing_b}', hit=None))
    if not missing_in:
        h = rng <= 0.1 and all(-0.1 <= v <= 0.3 for v in rt.values()) and all(0.02 <= v['delta'] <= 0.08 for v in rows_in.values())
        P.append(dict(line='2b: range c_rule <= 0.1; rule - table in [-0.1, 0.3]; delta in [0.02, 0.08]',
                      value=dict(range_c_rule=rng, rule_minus_table=rt, delta={n: v['delta'] for n, v in rows_in.items()}), hit=bool(h)))
    else:
        P.append(dict(line='2b', value=f'missing inputs {missing_in}', hit=None))
    out['predictions'] = dict(lines=P, hits=sum(1 for p in P if p['hit'] is True), misses=sum(1 for p in P if p['hit'] is False), n=len(P))
    # --- checks ---
    fits = [('U nominal', b['fit'], b['rec24'], b['rec120'])] + [(f'budget {k}', bu['rows'][k]['fit'], bu['rows'][k]['rec24'], bu['rows'][k]['rec120']) for k in rows_b] \
        + [(f'input {n}', r['fit'], r['rec24'], r['rec120']) for n, r in ins.items() if r is not None]
    chk2 = {lab: dict(EN24_le_B=bool(r24['EN'] <= f['B']), EN120_le_B=bool(r120['EN'] <= f['B']), j_plus_1_infeasible=f['j_plus_1_infeasible'],
                      mass_dev24=r24['mass_dev_max'], mass_dev120=r120['mass_dev_max']) for lab, f, r24, r120 in fits}
    out['checks'] = {
        '1_forwards': dict(pass_=True, rule='asserted in every forward (eval13: conservation 1e-11 sigma/24, 1e-10 sigma/120; min m >= 0; zero mass at the grid ends; '
                           'window edges inside the padded range); a failure stops the run',
                           max_mass_dev24=max(v['mass_dev24'] for v in chk2.values()), max_mass_dev120=max(v['mass_dev120'] for v in chk2.values())),
        '2_fitted_rules': dict(pass_=all(v['EN24_le_B'] and v['EN120_le_B'] and v['j_plus_1_infeasible'] for v in chk2.values()), rules=chk2),
        '3_monte_carlo': dict(pass_=bool(c['pass_']), z=Q['Q1']['z'], n_clipped=c['n_clipped']),
        '4_round10': (bu['rows'][f"{B_NOM:.10g}"].get('check4') if bu and f"{B_NOM:.10g}" in bu['rows'] else None),
        '5_nominal_fit_bitwise': (bu['rows'][f"{B_NOM:.10g}"].get('check5') if bu and f"{B_NOM:.10g}" in bu['rows'] else None)}
    with open(os.path.join(RES, 'round14.json'), 'w') as f: json.dump(out, f, indent=1)
    # --- log ---
    for k in Q: log(f"{k} {'YES' if Q[k]['answer'] else 'NO'}: " + json.dumps({kk: vv for kk, vv in Q[k].items() if kk != 'answer'})[:600])
    for k, v in rows_b.items():
        log(f"BUDGET B={v['B']:.3f}: q* {v['q_star']:.5f} e1 {v['e1']:.4f} e_eff {v['e_eff']:.4f} delta {v['delta']:+.4f} R({v['n']}, {v['j']}) EN120 {v['EN120']:.3f} "
            f"I_rule {v['I_rule']:.4f} I_SoD {v['I_SoD']:.4f} (M* {v['M_star']}, residual {v['SoD_residual']:+.3f}) dI {v['dI']:.4f} ips {v['ips_rule']:.4f}/{v['ips_SoD']:.4f} "
            f"const-threshold {v['I_const_threshold']:.4f} class-W {v['I_classW']:.4f} (u_cert {v['u_cert']:.4f}) class-W - rule {v['classW_minus_rule']:.4f}")
    for n, v in rows_in.items():
        log(f"INPUT {n}: R({v['n']}, {v['j']}) e_eff {v['e_eff']:.4f} delta {v['delta']:+.4f} I_rule {v['I_rule']:.4f} c_rule {v['c_rule']:.4f} rule - table {v['rule_minus_table']:+.4f} "
            f"rule - SoD {v['rule_minus_SoD']:.4f}")
    for p_ in P: log(f"PREDICTION {'HIT ' if p_['hit'] else ('MISS' if p_['hit'] is False else 'N/A ')} {p_['line']}: {json.dumps(p_['value'])[:500]}")
    log(f"PREDICTIONS {out['predictions']['hits']} hit, {out['predictions']['misses']} miss of {out['predictions']['n']}")
    for k, v in out['checks'].items(): log(f"CHECK {k}: {json.dumps(v)[:400]}")
    return out
