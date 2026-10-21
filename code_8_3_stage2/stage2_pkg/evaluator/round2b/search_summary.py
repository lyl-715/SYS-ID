"""Round 2b, Part B summary (no evaluation; reads the archive written by search_maxS.py). Spec reports_CN/plan_round2b_CN.md v3.2 §4.

  - the seven round-2a candidates: first evaluation of each NM run (x0 -> (a, chi) by the protocol map) vs round2a/results/recheck7.json
  - per NM run: best numerically feasible S in its archive, and the run's F at the result
  - top 10 numerically feasible evaluations of the whole archive; S_found and its parameters (from search_maxS/S_found.json)
  - T4 items: S_found in [500, 560]; p_1 > 0.4; Delta m_D below P5514's 0.701
  - copies the DENSE lines of search_maxS/search_maxS.log to search_maxS/dense_scan.log
Writes search_maxS/search_summary.log. Usage (from this directory): python3 search_summary.py"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE); OUT = os.path.join(HERE, 'search_maxS')
sys.path.insert(0, EV)
import density_recursion_v1 as dr

if __name__ == '__main__':
    sys.stdout = dr.Tee(os.path.join(OUT, 'search_summary.log'))
    arc = json.load(open(os.path.join(OUT, 'archive_all.json'))); recs = arc['records']; B = arc['B']
    sf = json.load(open(os.path.join(OUT, 'S_found.json')))
    starts = json.load(open(os.path.join(OUT, 'nm_starts.json')))
    rc7 = json.load(open(os.path.join(EV, 'round2a', 'results', 'recheck7.json')))
    print(f"=== search_summary.py; archive {len(recs)} evaluations (B = {B!r}); round2a/results/recheck7.json for the seven candidates")
    feas = lambda r: r['feas'] == '按数值判据可行'
    for st in starts:
        if st['kind'] != 'round2a candidate': continue
        r0 = [r for r in recs if r['stage'] == 'nm' and r['sid'] == st['sid'] and r['nfe'] == 1][0]; ref = rc7[st['origin']]
        print(f"OLD {st['origin']}: first NM evaluation S {r0['S']!r} E[N] {r0['EN']!r} ({r0['feas']}); round 2a recheck7 S {ref['S']!r} E[N] {ref['EN']!r}; "
              f"difference S {r0['S']-ref['S']!r}, E[N] {r0['EN']-ref['EN']!r}")
    print("RUN per NM run: evaluations, best numerically feasible S in the run's archive (evaluation number), its E[N]; F at the run's result")
    runs = {}
    for st in starts:
        rr = [r for r in recs if r['stage'] == 'nm' and r['sid'] == st['sid']]
        done = json.load(open(os.path.join(OUT, 'nm', f"done_{st['sid']}.json")))
        fz = [r for r in rr if feas(r)]; b = max(fz, key=lambda r: r['S']) if fz else None
        runs[st['sid']] = b
        print(f"RUN {st['sid']:16s} n {len(rr):3d} nfev {done['nfev']} | start S {rr[0]['S']:.4f} E[N] {rr[0]['EN']:.4f} ({rr[0]['feas']}) | best feasible "
              + (f"S {b['S']!r} (#{b['nfe']}) E[N] {b['EN']!r}" if b else 'none') + f" | F at result {-done['negF']!r}")
    kinds = {}
    for st in starts:
        b = runs[st['sid']]
        if b is not None: kinds.setdefault(st['kind'], []).append(b['S'])
    for k, v in kinds.items(): print(f"RUN-KIND {k}: {len(v)} runs with a feasible point; best feasible S max {max(v)!r}, median {float(np.median(v))!r}, min {min(v)!r}")
    fz = sorted([r for r in recs if feas(r)], key=lambda r: -r['S'])
    print("TOP10 numerically feasible evaluations by S")
    for r in fz[:10]:
        print(f"TOP10 {r['tag']:22s} S {r['S']!r} E[N] {r['EN']!r} (E[N] - B {r['EN']-B:+.6f}) dmD {r['dmD']!r} mE {r['mE']!r} "
              f"p ({', '.join(f'{v:.6f}' for v in r['p'])}) chi ({', '.join(f'{v:.6f}' for v in r['chi'])})")
    s = sf['S_found']
    print(f"S_FOUND {s['S']!r} ({s['tag']}); E[N] {s['EN']!r} (B - tau_N = {B-0.04!r}; margin {B-0.04-s['EN']!r}); dmD {s['dmD']!r}; mE {s['mE']!r}")
    print(f"S_FOUND p = ({', '.join(repr(v) for v in s['p'])}); a = ({', '.join(repr(v) for v in s['a'])}); chi mod 2 pi = ({', '.join(repr(v) for v in s['chi'])})")
    print(f"T4 S_found in [500, 560]: {500 <= s['S'] <= 560}; p_1 = {s['p'][0]!r} > 0.4: {s['p'][0] > 0.4}; dmD {s['dmD']!r} < 0.701 (P5514): {s['dmD'] < 0.701}")
    lines = [ln for ln in open(os.path.join(OUT, 'search_maxS.log')) if ln.startswith('DENSE')]
    open(os.path.join(OUT, 'dense_scan.log'), 'w').write(''.join(lines))
    print(f"DENSE {len(lines)} lines copied to search_maxS/dense_scan.log")
