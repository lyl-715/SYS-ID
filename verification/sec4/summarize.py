#!/usr/bin/env python3
"""Extract the summary of check_cert*.py runs (and the .meta file written by the run wrapper) as markdown.
usage: summarize.py [--overview] run_*.txt"""
import re, sys, os

def parse(fn):
    L = open(fn).read().splitlines()
    r = dict(file=fn, rounds=[], diffs=[], below=[], rule=None, time=None, MIN='1 sigma (hard-coded: FAC nodes)')
    cur = None
    for s in L:
        if m := re.match(r'grid sigma/(\d+): (\d+) nodes; candidates per side: (\d+) \((\d+) values', s):
            r['FAC'], r['n'], r['nvals'] = int(m[1]), int(m[2]), int(m[4])
        elif m := re.match(r'MIN = (\S+) sigma \((\d+) nodes\)', s):
            r['MIN'] = f"{m[1]} sigma ({m[2]} nodes)"
        elif m := re.match(r'constant SoD: E\[N\] = (\S+), log det J = (\S+),', s):
            r['B'], r['ld0'] = m[1], m[2]
        elif s.startswith('== tangent point'):
            cur = dict(diffs=[]); r['rounds'].append(cur)
        elif m := re.search(r'\(diff ([-+0-9.e]+)\)', s):
            cur['diffs'].append(float(m[1])); r['diffs'].append(float(m[1]))
        elif m := re.match(r'\s+best feasible table: log det J = (\S+) \((\S+)\), E\[N\] = (\S+)', s):
            cur['best'], cur['best_inc'], cur['EN'] = m[1], m[2], m[3]
        elif m := re.match(r'\s+upper bound for the class: (\S+) \((\S+)\);\s+gap = (\S+)', s):
            cur['UB'], cur['UB_inc'], cur['gap'] = m[1], m[2], m[3]
        elif m := re.search(r'below bound: (\w+)', s):
            r['below'].append(m[1] == 'True')
            if 'one-parameter' in s: r['rule'] = s.strip()
        elif m := re.match(r'time (\d+) s', s):
            r['time'] = int(m[1])
    meta = {}
    if os.path.exists(fn + '.meta'):
        for s in open(fn + '.meta'):
            k, _, v = s.partition(':'); meta[k.strip()] = v.strip()
    r['meta'] = meta
    r['err'] = os.path.getsize(fn + '.err') if os.path.exists(fn + '.err') else None
    return r

def md(r):
    o = [f"### `{r['meta'].get('cmd', r['file'])}`  →  `{r['file']}`", "",
         f"- grid sigma/{r.get('FAC')} ({r.get('n')} nodes); candidate set: {r.get('nvals')} values + None; MIN = {r['MIN']}",
         f"- constant SoD: E[N] = {r.get('B')}, log det J = {r.get('ld0')}", "",
         "| round | tangent point | best feasible table: log det J | E[N] | upper bound | gap | max abs(diff) in round |",
         "|---|---|---|---|---|---|---|"]
    for i, c in enumerate(r['rounds']):
        tp = 'constant SoD' if i == 0 else f'table of round {i}'
        md_ = f"{max(abs(d) for d in c['diffs']):.1e}" if c['diffs'] else '-'
        o.append(f"| {i+1} | {tp} | {c.get('best','(not reached)')} ({c.get('best_inc','')}) | {c.get('EN','')} | "
                 f"{c.get('UB','')} ({c.get('UB_inc','')}) | {c.get('gap','')} | {md_} |")
    o += ["", f"- largest abs(diff) between V* and the forward score, all {len(r['diffs'])} backward passes: "
              f"{max(abs(d) for d in r['diffs']):.1e}" if r['diffs'] else "- no diff values",
          f"- one-parameter rule: `{r['rule']}`" if r['rule'] else "- one-parameter rule: line not present",
          f"- 'below bound' entries: {len(r['below'])}, all True: {all(r['below']) if r['below'] else 'n/a'}",
          f"- wall time: {r['time']} s (script); wrapper: {r['meta'].get('wall_s','?')} s wall, "
          f"{r['meta'].get('cpu_user_s','?')} s user CPU, peak RSS {r['meta'].get('peak_rss_MB','?')} MB, "
          f"exit code {r['meta'].get('exit_code','?')}, stderr {r['err']} bytes", ""]
    return "\n".join(o)

def overview(runs):
    o = ["| run | grid | edge values | MIN (sigma) | const. SoD E[N] | const. SoD log det J | best table, round 3 | its E[N] "
         "| smallest bound | gap | max abs(diff) | one-parameter rule | all below bound | wall time (s) | peak RSS (MB) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        c = r['rounds'][-1]
        ub = min(float(x['UB']) for x in r['rounds'])
        rule = re.search(r'e = (\S+):\s+(\S+)', r['rule'])
        o.append(f"| `{r['file']}` | sigma/{r['FAC']} | {r['nvals']} + None | {r['MIN'].split(' sigma')[0]} | {r['B']} | {r['ld0']} | "
                 f"{c['best']} | {c['EN']} | {ub:.4f} | {c['gap']} | {max(abs(d) for d in r['diffs']):.1e} | "
                 f"{rule[2]} (e = {rule[1]}) | {all(r['below'])} | {r['time']} | {r['meta'].get('peak_rss_MB','?')} |")
    return "\n".join(o)

if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == '--overview':
        print(overview([parse(fn) for fn in a[1:]]))
    else:
        for fn in a:
            print(md(parse(fn)))
