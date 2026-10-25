"""Round 8b: the verdict of round 8 v3.1 §5 (computed by stats_r8.analyse inside stats_r8b.py; read here) and the scoring of
Q1-Q7 under DECISION S-0..S-7 of decisions_round8b.md (written before any run). Q1-Q6 use round 8's verdict_round8.subclaims
(imported unchanged; the readings are round 8's S-1..S-6); Q7 is this round's (S-7). The over-budget proportions are also
shown as "n/R observed" with Clopper-Pearson 95 % intervals next to the bootstrap intervals; the CP interval never enters the
verdict.

Usage (from this directory): python3 verdict_round8b.py   -> logs/verdict.log, results/verdict.json
                             python3 verdict_round8b.py --dir results/runs/e3_smoke --partB logs/engineering_repro_S.log  (test)
"""
import os, sys, json, re
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE); R8 = os.path.join(EV, 'round8')
for _p in (R8, HERE):
    if _p in sys.path: sys.path.remove(_p)
    sys.path.insert(0, _p)
_argv = sys.argv; sys.argv = sys.argv[:1]
import verdict_round8 as v8                               # round 8, unchanged (subclaims, score)
sys.argv = _argv
RES = os.path.join(HERE, 'results'); LOGS = os.path.join(HERE, 'logs')
B = v8.B


def subclaims(o, timing, wall):
    sub = v8.subclaims(o, timing, wall)
    sub['Q7'] = [(f"median primary work per replication {timing['primary_work_median']:.1f} s <= 110 s", timing['primary_work_median'] <= 110),
                 (f"Part B + Part C wall {wall/3600:.3f} h <= 3 h", wall <= 3*3600)]
    return sub

def main():
    official = '--dir' not in sys.argv
    res_dir = RES if official else os.path.abspath(sys.argv[sys.argv.index('--dir')+1])
    log_dir = LOGS if official else res_dir
    partB = os.path.join(LOGS, 'partB_pipeline.log') if official else os.path.abspath(sys.argv[sys.argv.index('--partB')+1])
    S = json.load(open(os.path.join(res_dir, 'stats_round8b.json'))); chk = json.load(open(os.path.join(RES, 'partA_checks.json')))
    m = re.findall(r'seq_pipeline_r8b\.py (?:run|repro) finished .*?; (\d+) replications in ([0-9.]+) s wall', open(partB).read())
    wallB = sum(float(x) for _, x in m)
    wall = wallB+S['timing']['partC_s']
    out = open(os.path.join(log_dir, 'verdict.log'), 'w')
    def P(x): print(x); out.write(x+'\n')
    o = S['P']; v = o['verdict']; c = o['tx_counts']
    P(f"V0 A3 required checks (round-8b acceptance): 1 {chk['A3_1']['pass']} 2 {chk['A3_2']['pass']} 3 {chk['A3_3']['pass']} 4 {chk['A3_4']['pass']}; "
      f"supporting {chk['supporting_pass']}")
    P(f"V1 [P] case {v['case']}; readings {v['readings']}; items {v['items']}; failure rates {v['failure_rates']}; max |bias|/RMSE "
      f"{v['max_bias_over_rmse']:.4f}; X = Delta D {v['X']!r} CI {v['X_ci']}; Y = median eps_1 {v['Y']!r}; Z = Pr{{E[N_2|T,theta_0] > B}} {v['Z']!r}")
    P(f"V1 [P] Z observed {c['EN2_gt_B']['n']}/{c['EN2_gt_B']['R']}: bootstrap 95 % {c['EN2_gt_B']['boot95']}"
      f"{' (degenerate)' if c['EN2_gt_B']['boot_degenerate'] else ''}, Clopper-Pearson 95 % {c['EN2_gt_B']['cp95']}; "
      f"Pr{{N_2 > B}} observed {c['N2_gt_B']['n']}/{c['N2_gt_B']['R']}: bootstrap {c['N2_gt_B']['boot95']}"
      f"{' (degenerate)' if c['N2_gt_B']['boot_degenerate'] else ''}, CP {c['N2_gt_B']['cp95']}")
    P(f"V2 Part B wall {wallB:.1f} s ({len(m)} run segment(s), {sum(int(n) for n, _ in m)} replications); Part C {S['timing']['partC_s']:.1f} s; "
      f"total {wall:.1f} s = {wall/3600:.3f} h; median primary work {S['timing']['primary_work_median']:.1f} s")
    res = dict(case=v['case'], readings=v['readings'], items=v['items'], X=v['X'], X_ci=v['X_ci'], Y=v['Y'], Z=v['Z'], Z_counts=c['EN2_gt_B'],
               N2_counts=c['N2_gt_B'], predictions={}, wall_s=wall, wallB_s=wallB)
    sub = subclaims(o, S['timing'], wall)
    for q in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'):
        sc = v8.score(sub[q])
        res['predictions'][q] = dict(score=sc, sub=[[t, bool(x)] for t, x in sub[q]])
        P(f"V3 {q} {sc}: " + '; '.join(f"{t} -> {'yes' if x else 'no'}" for t, x in sub[q]))
    json.dump(res, open(os.path.join(res_dir, 'verdict.json'), 'w'), indent=1, ensure_ascii=False)

if __name__ == '__main__':
    main()
