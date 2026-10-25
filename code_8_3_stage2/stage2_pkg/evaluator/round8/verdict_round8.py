"""Round 8: the verdict of spec §5 (from stats_r8.py, primary analysis P) and the scoring of Q1-Q7 under DECISION S-0..S-7
(written before Part B). The supplementary analysis S is shown next to each prediction, unscored.

Usage (from this directory): python3 verdict_round8.py   -> logs/verdict.log, results/verdict.json
"""
import os, sys, json, re
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'results'); LOGS = os.path.join(HERE, 'logs')
if '--dir' in sys.argv: RES = LOGS = os.path.abspath(sys.argv[sys.argv.index('--dir')+1])     # smoke test on the probe outputs
B = 316.8921168493721

def score(subs):
    n = sum(bool(v) for _, v in subs)
    return '对' if n == len(subs) else ('错' if n == 0 else '部分对')

def subclaims(o, timing, wall):
    st, tx, info, f, b = o['stage1'], o['tx'], o['info'], o['fail'], o['boot']
    e = st['eps1']; bor = [x for sc in ('seq', 'sod') for x in o['acc_all'][sc]['bias_over_rmse']]
    return {
        'Q1': [(f"median eps_1 {e['q50']:.4f} in [0.2, 0.4]", 0.2 <= e['q50'] <= 0.4), (f"q90 {e['q90']:.4f} <= 0.6", e['q90'] <= 0.6),
               (f"fraction > 0.5 {st['frac_eps1_gt_05']:.4f} <= 0.15", st['frac_eps1_gt_05'] <= 0.15),
               (f"stage-1 failure rate {f['stage1']:.4f} <= 0.02", f['stage1'] <= 0.02)],
        'Q2': [(f"|-log det M_1 - log det J_1| {info['diff_stage1']:.4f} <= 0.5", info['diff_stage1'] is not None and info['diff_stage1'] <= 0.5),
               (f"Hessian-FIM whitened {o['hfim']['whitened_norm']:.4f} <= 0.05", o['hfim']['whitened_norm'] <= 0.05)],
        'Q3': [(f"Pr{{E[N_2|T,theta_0] > B}} {tx['p_EN2_gt_B']:.4f} <= 0.10", tx['p_EN2_gt_B'] <= 0.10),
               (f"Pr{{N_2 > B}} {tx['p_N2_gt_B']:.4f} <= 0.25", tx['p_N2_gt_B'] <= 0.25),
               (f"b_seq {tx['b_seq']:.3f} <= 2B {2*B:.3f}", tx['b_seq'] <= 2*B)],
        'Q4': [(f"Delta D {o['dD']!r} in [2.5, 4]", o['dD'] is not None and 2.5 <= o['dD'] <= 4),
               (f"CI lower {b['dD'][0]!r} >= 1.5", b['dD'][0] is not None and b['dD'][0] >= 1.5)],
        'Q5': [(f"seq |.| {info['diff_seq']!r} <= 0.5", info['diff_seq'] is not None and info['diff_seq'] <= 0.5),
               (f"SoD+SoD |.| {info['diff_sod']!r} <= 0.5", info['diff_sod'] is not None and info['diff_sod'] <= 0.5)],
        'Q6': [(f"design failure rate {f['design']:.4f} <= 0.02", f['design'] <= 0.02),
               (f"estimation failure rates stage 1 {f['stage1']:.4f}, joint seq {f['joint_seq']:.4f}, joint SoD {f['joint_sod']:.4f} all <= 0.02",
                max(f['stage1'], f['joint_seq'], f['joint_sod']) <= 0.02),
               (f"max of 18 |bias|/RMSE {max(bor):.4f} <= 0.2", max(bor) <= 0.2)],
        'Q7': [(f"max primary-analysis work per replication {timing['primary_work_max']:.1f} s <= 60 s", timing['primary_work_max'] <= 60),
               (f"Part B + Part C wall {wall/3600:.3f} h <= 6 h", wall <= 6*3600)],
    }

def main():
    S = json.load(open(os.path.join(RES, 'stats_round8.json'))); chk = json.load(open(os.path.join(HERE, 'results', 'partA_checks.json')))
    logB = open(os.path.join(LOGS, 'partB_pipeline.log')).read()
    m = re.findall(r'replications in ([0-9.]+) s wall', logB)
    wallB = sum(float(x) for x in m)
    wall = wallB+S['timing']['partC_s']
    out = open(os.path.join(LOGS, 'verdict.log'), 'w')
    def P(x): print(x); out.write(x+'\n')
    P(f"V0 A3 required checks: 1 {chk['A3_1']['pass']} 2 {chk['A3_2']['pass']} 3 {chk['A3_3']['pass']} 4 {chk['A3_4']['pass']}; supporting {chk['supporting_pass']}")
    for an in ('P', 'S'):
        v = S[an]['verdict']
        P(f"V1 [{an}] {'binding' if an == 'P' else 'not binding (supplementary rule)'}: case {v['case']}; readings {v['readings']}; items {v['items']}; "
          f"X = Delta D {v['X']!r} CI {v['X_ci']}; Y = median eps_1 {v['Y']!r}; Z = Pr{{E[N_2|T,theta_0] > B}} {v['Z']!r}")
    P(f"V2 Part B wall {wallB:.1f} s ({len(m)} run segment(s)); Part C {S['timing']['partC_s']:.1f} s; total {wall:.1f} s = {wall/3600:.3f} h; "
      f"primary-only estimate (sum of primary work / 4 workers) {S['timing']['primary_work_sum']/4/3600:.3f} h")
    res = dict(case_P=S['P']['verdict']['case'], case_S=S['S']['verdict']['case'], predictions={}, wall_s=wall, wallB_s=wallB)
    subP = subclaims(S['P'], S['timing'], wall); subS = subclaims(S['S'], S['timing'], wall)
    for q in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'):
        sc = score(subP[q])
        res['predictions'][q] = dict(score=sc, sub=[[t, bool(v)] for t, v in subP[q]], supplementary=[[t, bool(v)] for t, v in subS[q]],
                                     supplementary_reading=score(subS[q]))
        P(f"V3 {q} [P, scored] {sc}: " + '; '.join(f"{t} -> {'yes' if v else 'no'}" for t, v in subP[q]))
        P(f"V3 {q} [S, unscored reading {score(subS[q])}]: " + '; '.join(f"{t} -> {'yes' if v else 'no'}" for t, v in subS[q]))
    json.dump(res, open(os.path.join(RES, 'verdict.json'), 'w'), indent=1, ensure_ascii=False)

if __name__ == '__main__':
    main()
