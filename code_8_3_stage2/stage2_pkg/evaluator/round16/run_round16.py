"""Round 16 driver (prompt reports_CN/prompt_round16_random_inputs_EN.md). Monte Carlo only; rounds 4-15 are imported read-only.

  python3 run_round16.py anchors   six anchors (seed 20261113) + check 1 (against Rounds 13/14; stop if it fails) + check 3 -> results/anchors.json
  python3 run_round16.py random    the 200 random inputs (seed 20261113), checkpointed                                     -> results/random.json
  python3 run_round16.py derived   4d: 50 phase-only (uniform power) + 50 power-only (zero phases)                          -> results/derived.json
  python3 run_round16.py floor     check 2: input U with the ten seeds 20261114 ... 20261123                               -> results/floor.json
  python3 run_round16.py summary   4a-4e, Q1-Q7, predictions, checks, figure                                     -> results/round16.json, fig_round16.png
Every input is one task of a Pool of 4 worker processes (prompt §8 runtime guide); within a task the runs are sequential.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
os.environ.setdefault('R13_INPUT', 'U')
import sys, json, time, platform, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import numpy as np
import scipy
from multiprocessing import Pool
import random16 as r16
s, K, B = r16.s, r16.K, r16.B
Stop, jdump = r16.Stop, r16.jdump
RES = os.path.join(HERE, 'results'); LOGS = os.path.join(HERE, 'logs')
for _d in (RES, LOGS): os.makedirs(_d, exist_ok=True)
WORKERS = 4
TOL_CHECK1 = 0.06
R14_JSON = os.path.join(EV, 'round14', 'results', 'round14.json')
R13_PARTA = os.path.join(EV, 'round13', 'results', 'partA_{}.json')
FROZEN = ('round4', 'round5', 'round6', 'round7', 'round8', 'round8b', 'round10', 'round11', 'round13', 'round14', 'round15')
FROZEN_FILES = ('config.json', 'density_recursion_v1.py', 'evaluator_fast.py')


class Tee:
    def __init__(self, path): self.f = open(path, 'a')
    def __call__(self, m=''): print(m, flush=True); self.f.write(m+'\n'); self.f.flush()

def versions():
    import numba
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__,
                workers=WORKERS, R=r16.R_MC, seed_noise=r16.SEED_NOISE, seed_inputs=r16.SEED_INPUTS)

def cleanup_frozen(out):
    """prompt §0: delete empty folders created under round4/ (and the other frozen rounds); git status on rounds 4-15 and frozen files."""
    removed = []
    for rd in FROZEN:
        root = os.path.join(EV, rd)
        if not os.path.isdir(root): continue
        for dp_, dns, fns in os.walk(root, topdown=False):
            if '__pycache__' in dp_: continue
            if not os.listdir(dp_): os.rmdir(dp_); removed.append(os.path.relpath(dp_, EV))
    st = subprocess.run(['git', 'status', '--porcelain', '--'] + [os.path.join(EV, r) for r in FROZEN if os.path.isdir(os.path.join(EV, r))]
                        + [os.path.join(EV, f) for f in FROZEN_FILES], capture_output=True, text=True, cwd=EV).stdout.strip()
    out(f"CLEANUP removed empty folders {removed}; git status on rounds 4-15 and the frozen files: {'clean' if not st else st}")
    return removed, st == ''


# --- tasks ---------------------------------------------------------------------------------------------------------------------
def build(spec):
    kind = spec[0]
    if kind == 'anchor': return r16.make_anchor(spec[1])
    _, name, p, chi = spec
    return r16.make_input(name, p, chi, kind)

def task(args):
    spec, seed = args
    inp = build(spec)
    return r16.process(inp, seed=seed)

def fmt(v, n=4): return '(' + ', '.join(f"{x:.{n}f}" for x in v) + ')'

def line(rec):
    d, der, fr, fs = rec['input'], rec['derived'], rec['rule'], rec['sod']
    return (f"IN {d['name']:>6s} seed {rec['seed']} p {fmt(d['p'], 3)} chi {fmt(d['chi'], 3)} range {d['range_sigma']:.1f} TV {d['tv_sigma']:.1f} "
            f"I_full {d['I_full']:.4f} | rule e_fit {der['e_fit']:.4f} pair {fr['fit']['pair']} ext {fr['extended']} I_rule {der['I_rule']:.4f} "
            f"c_rule {der['c_rule']:+.4f} | sod D_fit {der['D_fit']:.3f} pair {[round(x, 3) for x in fs['fit']['pair']]} ext {fs['extended']} "
            f"I_SoD {der['I_SoD']:.4f} c_sod {der['c_sod']:+.4f} | gain {der['gain']:.4f} | runs {der['n_runs']} {rec['wall_s']:.1f} s")

def run_tasks(specs, seeds, out, ckpt_path, label, every=10):
    """Pool of 4, ordered results, checkpoint every `every` completions; resumes from the checkpoint (skips names already done)."""
    res = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else dict(records={})
    res.setdefault('records', {}); res['versions'] = versions()
    todo = [(sp, sd) for sp, sd in zip(specs, seeds) if key(sp, sd) not in res['records']]
    out(f"{label}: {len(specs)} tasks, {len(specs)-len(todo)} already done, {len(todo)} to run on {WORKERS} workers")
    t0 = time.perf_counter(); n = 0
    with Pool(WORKERS) as pool:
        for (sp, sd), rec in zip(todo, pool.imap(task, todo)):
            res['records'][key(sp, sd)] = rec; n += 1
            out(line(rec))
            if n % every == 0: jdump(res, ckpt_path); out(f"  checkpoint {n}/{len(todo)} ({time.perf_counter()-t0:.0f} s)")
    res['wall_s'] = res.get('wall_s', 0.0)+time.perf_counter()-t0
    jdump(res, ckpt_path)
    return res

def key(spec, seed): return spec[1] if seed == r16.SEED_NOISE else f"{spec[1]}@{seed}"


# --- stages --------------------------------------------------------------------------------------------------------------------
def archives():
    r14 = json.load(open(R14_JSON))['input_table']; out = {}
    for n in r16.ANCHORS:
        a = json.load(open(R13_PARTA.format(n)))
        out[n] = dict(c_rule14=r14[n]['c_rule'], e_eff14=r14[n]['e_eff'], I_rule14=r14[n]['I_rule'], EN120_rule14=r14[n]['EN120'],
                      I_full13=a['derived']['I_full'], c_sod13=a['derived']['c_sod'], I_SoD13=a['derived']['I_SoD'],
                      M_star_sigma13=a['matched']['M_star_sigma'], EN120_sod13=a['matched']['EN120'])
    return out

def stage_anchors(out):
    path = os.path.join(RES, 'anchors.json')
    specs = [('anchor', n) for n in r16.ANCHORS]
    res = run_tasks(specs, [r16.SEED_NOISE]*len(specs), out, path, 'anchors')
    arc = archives(); rows = {}; ok = True
    for n in r16.ANCHORS:
        rec = res['records'][n]; d = rec['derived']; a = arc[n]
        dr_ = d['c_rule']-a['c_rule14']; ds = d['c_sod']-a['c_sod13']
        p = abs(dr_) <= TOL_CHECK1 and abs(ds) <= TOL_CHECK1; ok &= p
        rows[n] = dict(c_rule=d['c_rule'], c_rule14=a['c_rule14'], diff_rule=dr_, c_sod=d['c_sod'], c_sod13=a['c_sod13'], diff_sod=ds,
                       e_fit=d['e_fit'], e_eff14=a['e_eff14'], diff_e=d['e_fit']-a['e_eff14'], D_fit=d['D_fit'], M_star_sigma13=a['M_star_sigma13'],
                       diff_D=d['D_fit']-a['M_star_sigma13'], I_full=rec['input']['I_full'], I_full13=a['I_full13'],
                       diff_I_full=rec['input']['I_full']-a['I_full13'], I_rule=d['I_rule'], I_rule14=a['I_rule14'], I_SoD=d['I_SoD'],
                       I_SoD13=a['I_SoD13'], pass_=p)
        out(f"CHECK1 {n:>6s}: c_rule {d['c_rule']:+.4f} vs R14 {a['c_rule14']:+.4f} diff {dr_:+.4f}; c_sod {d['c_sod']:+.4f} vs R13 {a['c_sod13']:+.4f} "
            f"diff {ds:+.4f}; e_fit {d['e_fit']:.4f} vs e_eff {a['e_eff14']:.4f} ({d['e_fit']-a['e_eff14']:+.4f}); D_fit/sigma {d['D_fit']:.3f} vs "
            f"M*/24 {a['M_star_sigma13']:.3f} ({d['D_fit']-a['M_star_sigma13']:+.3f}); I_full diff {rec['input']['I_full']-a['I_full13']:+.1e} -> "
            f"{'PASS' if p else 'FAIL'}")
    out(f"CHECK1 max |diff| c_rule {max(abs(v['diff_rule']) for v in rows.values()):.4f}, c_sod {max(abs(v['diff_sod']) for v in rows.values()):.4f} "
        f"(tolerance {TOL_CHECK1}) -> {'PASS' if ok else 'FAIL'}")
    # check 3: pair LO - HI
    c = {n: res['records'][n]['derived'] for n in ('LO', 'HI')}
    pr = dict(rule=c['LO']['c_rule']-c['HI']['c_rule'], rule14=arc['LO']['c_rule14']-arc['HI']['c_rule14'],
              sod=c['LO']['c_sod']-c['HI']['c_sod'], sod13=arc['LO']['c_sod13']-arc['HI']['c_sod13'])
    pr['diff_rule'] = pr['rule']-pr['rule14']; pr['diff_sod'] = pr['sod']-pr['sod13']
    out(f"CHECK3 c_rule(LO) - c_rule(HI) {pr['rule']:+.4f} vs R14 {pr['rule14']:+.4f} (diff {pr['diff_rule']:+.4f}); "
        f"c_sod(LO) - c_sod(HI) {pr['sod']:+.4f} vs R13 {pr['sod13']:+.4f} (diff {pr['diff_sod']:+.4f})")
    res['check1'] = dict(rows=rows, tol=TOL_CHECK1, pass_=bool(ok)); res['check3'] = pr; res['archives'] = arc
    jdump(res, path)
    if not ok: raise Stop("check 1 failed: the Monte Carlo route does not reproduce the archived anchors within 0.06 nat (prompt §3)")

def random_specs():
    acc, rej, draws = r16.draw_random_set()
    return [('random', f"r{i:03d}", p, chi) for i, (p, chi) in enumerate(acc)], rej, draws

def stage_random(out):
    path = os.path.join(RES, 'random.json')
    specs, rej, draws = random_specs()
    out(f"RANDOM set: {draws} draws, {rej} rejected (min p < {r16.P_MIN}), {len(specs)} accepted")
    res = run_tasks(specs, [r16.SEED_NOISE]*len(specs), out, path, 'random')
    res['draws'] = draws; res['rejected'] = rej; res['accepted'] = len(specs); jdump(res, path)

def stage_derived(out):
    path = os.path.join(RES, 'derived.json')
    specs, _, _ = random_specs(); n = r16.N_DERIVED
    ph = [('phase_only', f"ph{i:03d}", np.full(4, 0.25), sp[3]) for i, sp in enumerate(specs[:n])]
    pw = [('power_only', f"pw{i:03d}", sp[2], np.zeros(4)) for i, sp in enumerate(specs[:n])]
    run_tasks(ph+pw, [r16.SEED_NOISE]*(2*n), out, path, 'derived')

def stage_floor(out):
    path = os.path.join(RES, 'floor.json')
    specs = [('anchor', 'U')]*len(r16.SEEDS_FLOOR)
    res = run_tasks(specs, list(r16.SEEDS_FLOOR), out, path, 'floor')
    vals = {k: [res['records'][f"U@{sd}"]['derived'][k] for sd in r16.SEEDS_FLOOR] for k in ('c_rule', 'c_sod', 'e_fit', 'D_fit', 'gain')}
    res['sd'] = {k: float(np.std(v, ddof=1)) for k, v in vals.items()}; res['mean'] = {k: float(np.mean(v)) for k, v in vals.items()}
    out("CHECK2 noise floor, input U, ten seeds 20261114..20261123 (no common random numbers between seeds): sd " +
        ', '.join(f"{k} {v:.4f}" for k, v in res['sd'].items()))
    jdump(res, path)


if __name__ == '__main__':
    stage = sys.argv[1]
    out = Tee(os.path.join(LOGS, f'{stage}.log'))
    out(f"=== ROUND 16 stage {stage} start {time.strftime('%Y-%m-%d %H:%M:%S')} versions {versions()}")
    t0 = time.perf_counter(); code = 0
    try:
        if stage == 'anchors': stage_anchors(out)
        elif stage == 'random': stage_random(out)
        elif stage == 'derived': stage_derived(out)
        elif stage == 'floor': stage_floor(out)
        elif stage == 'summary':
            import summary16; summary16.main(out)
        else: raise SystemExit(f"unknown stage {stage}")
    except Stop as e:
        out(f"STOP {e}"); code = 2
    cleanup_frozen(out)
    out(f"=== ROUND 16 stage {stage} done in {time.perf_counter()-t0:.1f} s (exit {code})")
    sys.exit(code)
