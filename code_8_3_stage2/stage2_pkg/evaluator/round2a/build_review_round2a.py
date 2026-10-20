"""Review package for round 2a, in the layout of review_8_5_stage2.zip: one top folder review_pkg/ that mirrors stage2_pkg/, with
REVIEW_GUIDE.md at the top. Contents: evaluator/round2a/ in full (scripts, config, predictions, report, table, logs, results,
figures; not the untracked *.stdout copies or __pycache__); the cited lines of evaluator/logs/analysis.log (197-213, tau_N and
tau_E), run_all.log (1021, B) and speedup_regression.log (65, B) as excerpts that keep their line numbers; the spec v4.2 and the
note it cites; and, unchanged, the files the round-2a scripts import or cite, for reading. The package is for review;
reproduction needs the full repository checkout, not the zip alone.
Usage (from this directory): python3 build_review_round2a.py  ->  <repo root>/review_round2a.zip"""
import os, zipfile

HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE); PKG = os.path.dirname(EV)
ROOT = os.path.dirname(os.path.dirname(PKG)); OUT = os.path.join(ROOT, 'review_round2a.zip')
EXCERPTS = [('evaluator/logs/analysis.log', 197, 213, 'tau_N (STAB ... EN) and tau_E (STAB ... mE), version B, sigma/12 vs sigma/24'),
            ('evaluator/logs/run_all.log', 1021, 1021, 'B = E[N](u_unif), version B, n = 24'),
            ('evaluator/logs/speedup_regression.log', 65, 65, 'B reproduced bitwise by evaluator_fast.py')]
DEPS = ['mc83.py', 'mc85_scan.py', 'stage2_common.py', 'stage2_local.py', 'wk_d20_10000.npy',
        'evaluator/density_recursion_v1.py', 'evaluator/evaluator_fast.py', 'evaluator/config.json', 'evaluator/config_fast.json',
        'evaluator/results/rec_B_U_n24_m8.json',
        'bound/bound_common.py', 'bound/bound_report.md', 'bound/bound_inner.log',
        'stage2/final_batch1.json', 'stage2/stage2_final.log', 'stage2/stage2_report.md',
        'stage3/unconstrained.json', 'stage3/stage3_unconstrained.log']
SPEC = ['reports_CN/plan_recheck_bounds_CN.md', 'reports_CN/global_bound_notes_CN.md']

with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(os.path.join(HERE, 'REVIEW_GUIDE.md'), 'review_pkg/REVIEW_GUIDE.md')
    for d, dirs, fs in os.walk(HERE):
        dirs[:] = [x for x in dirs if x != '__pycache__']
        for f in sorted(fs):
            if f.endswith('.stdout') or f == '.gitignore': continue
            p = os.path.join(d, f); zf.write(p, 'review_pkg/evaluator/round2a/' + os.path.relpath(p, HERE))
    for path, a, b, what in EXCERPTS:
        lines = open(os.path.join(PKG, path)).read().splitlines()
        body = f"# excerpt of {path}, lines {a}-{b} ({what}); each line is 'line number: original text', verbatim\n"
        body += ''.join(f"{i}: {lines[i-1]}\n" for i in range(a, b+1))
        name = os.path.basename(path).replace('.log', f'_log_lines_{a}-{b}.txt' if a != b else f'_log_line_{a}.txt')
        zf.writestr(f'review_pkg/evaluator/log_excerpts/{name}', body)
    for f in DEPS: zf.write(os.path.join(PKG, f), f'review_pkg/{f}')
    for f in SPEC: zf.write(os.path.join(ROOT, f), f'review_pkg/{f}')
    n = len(zf.namelist())
print(f"{OUT}: {n} files, {os.path.getsize(OUT)/1e6:.2f} MB")
