"""Round 2a, A1-A2: recheck of the seven candidates with the accepted fast evaluator (spec reports_CN/plan_recheck_bounds_CN.md
v4.2, §A2-§A4; frozen rules in config_recheck.json).

evaluator_fast.evaluate_fast (version B, n = 24, margin 8 Delta), unchanged. The candidates' a_m, chi_m are added to the
evaluator's in-memory input table only (density_recursion_v1.CFG['inputs']); no file of the evaluator is modified. U is evaluated
first with the same evaluator; its E[N] must equal the frozen budget B bitwise. Per candidate: E[N] (discrete Phat; exact-P
diagnostic too), S, J (.npy), Delta m_D vs U, nine lambda_i (J v = lambda J_U v), m_E = lambda_1; then the A4 table with the
feasibility rule (B +- tau_N, budget relative to the uniform input), the lambda0 rule (+- tau_E) and (rec - MC)/se against the
R = 4000 paired se. Writes logs/recheck7.log (one line per number), results/J_*.npy, recheck7_table.md, results/recheck7.json.
Usage (from this directory): python3 recheck7.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
sys.path.insert(0, EV)
import numpy as np
import density_recursion_v1 as dr
import evaluator_fast as ef

CFG = json.load(open(os.path.join(HERE, 'config_recheck.json')))
B = CFG['budget_B']['value']; TN = CFG['tau_N']['value']; TE = CFG['tau_E']['value']
LOGS = os.path.join(HERE, 'logs'); RES = os.path.join(HERE, 'results'); os.makedirs(LOGS, exist_ok=True); os.makedirs(RES, exist_ok=True)


def feasibility(en):
    if en <= B-TN: return '按数值判据可行'
    if en >= B+TN: return '按数值判据不可行'
    return '边界未定'

def lambda0_verdict(me, floor):
    if floor is None: return '不适用（无 λ0 约束）'
    if me >= floor+TE: return f'按数值判据满足 λ0 = {floor}'
    if me <= floor-TE: return f'按数值判据不满足 λ0 = {floor}'
    return f'未定（λ0 = {floor}）'


if __name__ == '__main__':
    sys.stdout = dr.Tee(os.path.join(LOGS, 'recheck7.log'))
    print(f"=== recheck7.py (round 2a A1-A2); spec {CFG['spec']}; evaluator {CFG['evaluator']}; B = {B!r}, tau_N = {TN}, tau_E = {TE}")
    for c in CFG['candidates']: dr.CFG['inputs'][c['id']] = dict(a=c['a'], chi=c['chi'])      # in memory only
    ef._warmup()
    recU, JU, rU = ef.evaluate_fast('U', 24)
    print(f"A1 U EN {recU['EN']!r}; equals frozen B bitwise: {recU['EN'] == B}; assertion failures {int((~rU['assert_ok']).sum())}")
    assert recU['EN'] == B
    np.save(os.path.join(RES, 'J_UNIF.npy'), JU)
    rows = [dict(id='UNIF', source='reference (spec §A4)', dmD=0.0, mc='0（定义）', z='—', EN=recU['EN'], feas='按定义合格',
                 mE=1.0, l0='—', S=recU['S'])]
    for k in ('EN', 'EN_P', 'S', 'logdet'): print(f"A1 UNIF {k} {float(recU[k])!r}")
    out = {'UNIF': {k: recU[k] for k in ('EN', 'EN_P', 'S', 'logdet')}}
    for c in CFG['candidates']:
        cid = c['id']
        rec, J, r = ef.evaluate_fast(cid, 24)
        dr.relative(rec, J, JU)
        np.save(os.path.join(RES, f'J_{cid}.npy'), J)
        nfail = int((~r['assert_ok']).sum())
        for k in ('EN', 'EN_P', 'S', 'logdet', 'dmD', 'mE', 'mass_dev_max', 'min_m'): print(f"A1 {cid} {k} {float(rec[k])!r}")
        for i, v in enumerate(rec['lam']): print(f"A1 {cid} lam{i+1} {v!r}")
        print(f"A1 {cid} assertion_failures {nfail}")
        assert nfail == 0
        mc = c['mc']
        if mc['paired_se'] is not None:
            z = (rec['dmD']-mc['dmD'])/mc['paired_se']; zs = f'{z:+.2f}'; mcs = f"{mc['dmD']:+.4f} ± {mc['paired_se']:.4f}"
            print(f"A2 {cid} MC dmD {mc['dmD']!r} paired_se {mc['paired_se']!r} z {z!r}")
        else:
            zs = '未定'; mcs = f"{mc['dmD']:+.4f}（R = 2000，无 R = 4000 配对 se）"
            print(f"A2 {cid} MC dmD {mc['dmD']!r} (R = 2000) paired_se none at R = 4000; z 未定")
        fe = feasibility(rec['EN']); l0 = lambda0_verdict(rec['mE'], c['floor_lambda0'])
        print(f"A2 {cid} E[N] - B {rec['EN']-B!r}; feasibility {fe}")
        print(f"A2 {cid} lambda0 verdict {l0}" + ('' if c['floor_lambda0'] is None else f"; m_E - lambda0 {rec['mE']-c['floor_lambda0']!r}"))
        rows.append(dict(id=cid, source=c['source_params'].replace('code_8_3_stage2/stage2_pkg/', ''), dmD=rec['dmD'], mc=mcs, z=zs,
                         EN=rec['EN'], feas=fe, mE=rec['mE'], l0=l0, S=rec['S']))
        out[cid] = {k: rec[k] for k in ('EN', 'EN_P', 'S', 'logdet', 'dmD', 'mE', 'lam')}
        out[cid].update(feasibility=fe, lambda0=l0, z=zs)
    json.dump(out, open(os.path.join(RES, 'recheck7.json'), 'w'), indent=1, ensure_ascii=False)
    hdr = ('| ID | 来源 | Δm_D（rec） | MC 值 ± 配对 se（R = 4000） | (rec − MC)/se | E[N] | 可行性（按数值判据，B ± τ_N） | m_E | λ0 判定（± τ_E） | S |\n'
           '|---|---|---|---|---|---|---|---|---|---|\n')
    body = ''.join(f"| {r['id']} | {r['source']} | {r['dmD']:+.4f} | {r['mc']} | {r['z']} | {r['EN']:.4f} | {r['feas']} | {r['mE']:.4f} | "
                   f"{r['l0']} | {r['S']:.3f} |\n" for r in rows[1:]+rows[:1])
    tab = (f"B = E[N](u_unif) = {B:.4f}（同评价器 B/n=24），τ_N = {TN}，τ_E = {TE}；评价器 evaluator_fast.py（B，n = 24）。"
           f"来源：logs/recheck7.log（A1、A2 行）。\n\n" + hdr + body)
    open(os.path.join(HERE, 'recheck7_table.md'), 'w').write(tab)
    print(tab)
