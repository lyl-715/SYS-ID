"""Round 18 (second session): compare a re-run of stages A and B of the committed code (e85a3f0) with the committed results.

Usage: python3 rerun_check18.py <round18 directory of the re-run>   (a worktree at e85a3f0 whose round18/results and logs were removed
before running stages A and B there). Criteria: JSON - every field except wall_s / wall_s_sum equal in value and type; .npy - arrays
and file bytes; partB_estimates.npz - every array (the zip container carries a write time, so its bytes are not compared); logs -
every line after removing wall times."""
import json, os, glob, re, sys
import numpy as np

new = sys.argv[1]
ref = os.path.dirname(os.path.abspath(__file__))
TIME = ('wall_s', 'wall_s_sum')

def cmp(a, b, path=''):
    diffs = []
    if isinstance(a, dict):
        if set(a) != set(b): diffs.append((path, 'keys', sorted(set(a) ^ set(b))))
        for k in a:
            if k in TIME: continue
            if k in b: diffs += cmp(a[k], b[k], path+'.'+k)
    elif isinstance(a, list):
        if len(a) != len(b): return [(path, 'len')]
        for i, (x, y) in enumerate(zip(a, b)): diffs += cmp(x, y, f'{path}[{i}]')
    else:
        if not (a == b and type(a) == type(b)): diffs.append((path, a, b))
    return diffs

ok = True
for f in ('partA.json', 'partB.json'):
    a = json.load(open(f'{ref}/results/{f}')); b = json.load(open(f'{new}/results/{f}'))
    d = cmp(a, b); ok &= not d
    print(f, 'differences outside wall_s / wall_s_sum:', len(d), d[:5])
    print('  wall_s committed / re-run:', round(a['wall_s'], 1), round(b['wall_s'], 1))
for f in sorted(glob.glob(f'{ref}/results/*.npy')):
    g = os.path.join(new, 'results', os.path.basename(f))
    x = np.load(f); y = np.load(g)
    eq = bool(np.array_equal(x, y) and x.dtype == y.dtype); eb = open(f, 'rb').read() == open(g, 'rb').read(); ok &= eq and eb
    print(os.path.basename(f), 'arrays bitwise equal', eq, '| file bytes equal', eb)
x = np.load(f'{ref}/results/partB_estimates.npz'); y = np.load(f'{new}/results/partB_estimates.npz')
eq = sorted(x.files) == sorted(y.files) and all(np.array_equal(x[k], y[k]) and x[k].dtype == y[k].dtype for k in x.files); ok &= eq
print('partB_estimates.npz:', len(x.files), 'arrays; keys and all arrays bitwise equal', eq)
strip = lambda L: [re.sub(r'(done in|wall|\() ?[0-9.]+ s', '<t>', l) for l in L]
for st in 'AB':
    l1 = open(f'{ref}/logs/{st}.log').read().splitlines(); l2 = open(f'{new}/logs/{st}.log').read().splitlines()
    dl = [(i+1, p, q) for i, (p, q) in enumerate(zip(strip(l1), strip(l2))) if p != q]; ok &= (not dl and len(l1) == len(l2))
    print(f'log {st}: {len(l1)} / {len(l2)} lines; differing lines after removing wall times: {len(dl)}', [d[0] for d in dl])
print(f"CHECK re-run of stages A and B reproduces the committed results exactly (outside wall times): {ok} -> {'PASS' if ok else 'FAIL'}")
