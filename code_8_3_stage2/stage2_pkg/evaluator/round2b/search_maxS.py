"""Round 2b, Part B: in-family max-S search with the fast evaluator (spec reports_CN/plan_round2b_CN.md v3.2, frozen, §4).

Every number here is an evaluated value of the accepted evaluator evaluator_fast.py (version B, n = 24, margin 8 Delta),
unchanged; the inputs are added to its in-memory input table only (density_recursion_v1.CFG['inputs'], as round2a/recheck7.py
does). S_found is "按数值判据可行的已评估候选中最大的 S 评价值": neither a lower bound of the continuous problem (feasibility and
evaluation error are not certified) nor the family maximum.

Guidance objective (search only): F(u) = S(u) - 20 max{E[N](u) - B, 0}. Final selection is separate: the round-2a feasibility rule
(round2a/recheck7.feasibility, reused: E[N] <= B - tau_N 按数值判据可行, >= B + tau_N 不可行, else 边界未定), B and tau_N from the frozen
round2a/config_recheck.json. Every evaluation is archived (u, S, E[N], Delta m_D, m_E, ...), duplicates included (不去重).

Stages (each resumable from its checkpoint files in search_maxS/; a killed process resumes, it does not restart):
  grid   6^4 = 1296 phase points at uniform power, chi_m in {0, 60, ..., 300} deg (itertools.product order of mc85_scan.py)
  nm     47 Nelder-Mead runs with the stage-2 protocol of stage2_pkg/stage2_local.py (parametrisation, initial simplex, scipy
         options reused from that file's source): starts = 20 grid points with the highest S among the numerically feasible
         (topped up by F if fewer than 20) + the seven round-2a candidates + 20 random (seed 20261022)
  dense  coordinate-wise scan around S_found, centre fixed: 48 phase points + <= 40 power points; no re-centring
  summary  S_found, its parameters, counts, runtimes
Usage (from this directory):  nohup python3 search_maxS.py all > search_maxS/run.stdout 2>&1 &"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, itertools, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE); PKG = os.path.dirname(EV)
sys.path.insert(0, EV); sys.path.insert(0, os.path.join(EV, 'round2a'))
import numpy as np
import multiprocessing as mp
from scipy.optimize import minimize
import density_recursion_v1 as dr
import evaluator_fast as ef
import recheck7                                                    # round-2a feasibility rule (function reused, not copied)

TWO_PI = 2*np.pi
OUT = os.path.join(HERE, 'search_maxS'); os.makedirs(OUT, exist_ok=True)
CFG2A = json.load(open(os.path.join(EV, 'round2a', 'config_recheck.json')))
B = CFG2A['budget_B']['value']; TAU_N = CFG2A['tau_N']['value']
assert B == recheck7.B and TAU_N == recheck7.TN
MU = 20.0                                                         # spec §4: penalty 20 per send over budget
N_GRID_STARTS, N_RAND, SEED_RAND = 20, 20, 20261022
PHASE_DEG = list(range(1, 7)); POWER_T = [0.002, 0.004, 0.006, 0.008, 0.010]
CAP = 1296+47*150+88                                              # spec §4: 8434

# --- stage-2 protocol, reused from the source of stage2_common.py / stage2_local.py (same text, exec'd) -----------------
_s2c = open(os.path.join(PKG, 'stage2_common.py')).read()
exec('def p_of_z(z):'+_s2c.split('def p_of_z(z):')[1].split('\ndef ')[0])
_s2l = open(os.path.join(PKG, 'stage2_local.py')).read()
for _f in ('x_to_input', 'input_to_x'):
    exec([ln for ln in _s2l.splitlines() if ln.startswith(f'def {_f}(')][0])
_line = [ln for ln in _s2l.splitlines() if ln.startswith('MU_B, MU_E, H_Z, H_CHI, MAXFEV = ')][0]
_, _, H_Z, H_CHI, MAXFEV = eval(_line.split('=', 1)[1])
_sim_src = [ln.strip() for ln in _s2l.splitlines() if ln.strip().startswith('sim = np.vstack(')][0]
_min_src = [ln.strip() for ln in _s2l.splitlines() if ln.strip().startswith('res = minimize(f, x0, method=')][0]
assert (H_Z, H_CHI, MAXFEV) == (0.25, 0.25, 150) and "xatol=1e-4, fatol=1e-5" in _min_src and "'Nelder-Mead'" in _min_src


def F_of(S, EN): return S-MU*max(EN-B, 0.0)


# --- one evaluation (worker side) ----------------------------------------------------------------------------------------------
_JU = None

def _init_worker(JU):
    global _JU
    _JU = JU
    ef._warmup()

def evaluate_u(a, chi, tag):
    """evaluator_fast (B, n = 24) on the input (a, chi); chi reduced mod 2 pi by the evaluator's input_arrays."""
    name = f'r2b:{tag}'
    a = [float(v) for v in a]; chi = [float(v) for v in chi]
    dr.CFG['inputs'][name] = dict(a=a, chi=chi)
    try:
        t0 = time.time(); rec, J, r = ef.evaluate_fast(name, 24); dr.relative(rec, J, _JU); wall = time.time()-t0
    finally:
        del dr.CFG['inputs'][name]
    nfail = int((~r['assert_ok']).sum())
    if nfail: raise ef.AssertionStop(f'{tag}: {nfail} window assertion failures')
    S, EN = float(rec['S']), float(rec['EN'])
    return dict(tag=tag, a=a, p=[v*v for v in a], chi=[float(v) for v in np.mod(chi, TWO_PI)], S=S, EN=EN, EN_P=float(rec['EN_P']),
                dmD=float(rec['dmD']), mE=float(rec['mE']), logdet=float(rec['logdet']), F=F_of(S, EN),
                feas=recheck7.feasibility(EN), runtime_s=float(rec['runtime_s']), wall_s=wall, pid=os.getpid())

def _eval_task(t):
    a, chi, tag, extra = t
    rec = evaluate_u(a, chi, tag); rec.update(extra); return rec


# --- helpers ----------------------------------------------------------------------------------------------------------------------
def read_jsonl(path):
    if not os.path.exists(path): return []
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            try: out.append(json.loads(ln))
            except json.JSONDecodeError: pass                       # a line cut by a kill: dropped, re-evaluated
    return out

def clean_jsonl(path):
    """rewrite a checkpoint file with its complete lines only (a line cut by a kill is dropped and re-evaluated)"""
    recs = read_jsonl(path)
    if os.path.exists(path):
        with open(path+'.tmp', 'w') as f: f.write(''.join(json.dumps(r)+'\n' for r in recs))
        os.replace(path+'.tmp', path)
    return recs

def append_jsonl(path, rec):
    with open(path, 'a') as f: f.write(json.dumps(rec)+'\n'); f.flush(); os.fsync(f.fileno())

def is_feasible(r): return r['feas'] == '按数值判据可行'

def fmt_rec(r):
    return (f"S {r['S']!r} E[N] {r['EN']!r} (E[N] - B {r['EN']-B:+.6f}; {r['feas']}) F {r['F']!r} dmD {r['dmD']!r} mE {r['mE']!r} "
            f"p ({', '.join(f'{v:.6f}' for v in r['p'])}) chi ({', '.join(f'{v:.6f}' for v in r['chi'])})")

def run_pool(tasks, path, log):
    """evaluate the tasks not yet in the checkpoint file (keyed by tag), 4 workers, one line per evaluation appended on return."""
    done = {r['tag'] for r in clean_jsonl(path)}; todo = [t for t in tasks if t[2] not in done]
    log(f"  {len(tasks)} tasks, {len(tasks)-len(todo)} already in {os.path.relpath(path, HERE)}, {len(todo)} to evaluate")
    if todo:
        t0 = time.time(); n = 0
        with mp.get_context('fork').Pool(4, initializer=_init_worker, initargs=(JU,)) as pl:
            for rec in pl.imap_unordered(_eval_task, todo, chunksize=1):
                append_jsonl(path, rec); n += 1
                if n % 100 == 0 or n == len(todo):
                    log(f"  progress {n}/{len(todo)} evaluations, wall {time.time()-t0:.0f} s")
    byt = {r['tag']: r for r in read_jsonl(path)}
    return [byt[t[2]] for t in tasks]


# --- stage: grid ------------------------------------------------------------------------------------------------------------------
def grid_tasks():
    g = np.arange(6)*np.pi/3                                       # as in mc85_scan.py (B) phase scan
    return [([0.5]*4, list(c), f'grid{i:04d}', dict(stage='grid', idx=i, deg=[int(round(v*180/np.pi)) for v in c]))
            for i, c in enumerate(itertools.product(g, repeat=4))]

def stage_grid(log):
    log(f"GRID 6^4 phase grid at uniform power (a_m = 0.5), chi_m in {{0, 60, ..., 300}} deg")
    recs = run_pool(grid_tasks(), os.path.join(OUT, 'eval_grid.jsonl'), log)
    nf = sum(is_feasible(r) for r in recs); nu = sum(r['feas'] == '边界未定' for r in recs)
    log(f"GRID evaluations {len(recs)}; 按数值判据可行 {nf}; 边界未定 {nu}; 不可行 {len(recs)-nf-nu}")
    bs = max(recs, key=lambda r: r['S']); bf = max([r for r in recs if is_feasible(r)], key=lambda r: r['S'])
    log(f"GRID largest S (any feasibility) {bs['tag']} deg {bs['deg']}: {fmt_rec(bs)}")
    log(f"GRID largest S among 按数值判据可行 {bf['tag']} deg {bf['deg']}: {fmt_rec(bf)}")
    return recs


# --- stage: Nelder-Mead -------------------------------------------------------------------------------------------------------------
def make_starts(grid, log):
    feas = sorted([r for r in grid if is_feasible(r)], key=lambda r: (-r['S'], r['idx']))
    sel = feas[:N_GRID_STARTS]; how = ['top S among 按数值判据可行']*len(sel)
    if len(sel) < N_GRID_STARTS:
        rest = sorted([r for r in grid if r not in sel], key=lambda r: (-r['F'], r['idx']))
        add = rest[:N_GRID_STARTS-len(sel)]; sel += add; how += ['top-up by F']*len(add)
    starts = []
    for j, (r, h) in enumerate(zip(sel, how)):
        starts.append(dict(sid=f'G{j+1:02d}', kind='grid', origin=r['tag'], how=h, p=[0.25]*4, chi=r['chi'], S0=r['S'], EN0=r['EN']))
    for c in CFG2A['candidates']:
        starts.append(dict(sid=f"C-{c['id']}", kind='round2a candidate', origin=c['id'], how='config_recheck.json', p=c['p'], chi=c['chi']))
    rng = np.random.default_rng(SEED_RAND)
    P = rng.dirichlet(np.ones(4), N_RAND); CH = rng.uniform(0.0, TWO_PI, (N_RAND, 4))      # powers first, then phases
    for j in range(N_RAND):
        starts.append(dict(sid=f'R{j+1:02d}', kind='random', origin=f'seed {SEED_RAND} draw {j}', how='Dirichlet(1,1,1,1) powers, U[0,2pi) phases',
                           p=P[j].tolist(), chi=CH[j].tolist()))
    for s_ in starts: s_['x0'] = input_to_x(s_['p'], s_['chi']).tolist()
    for s_ in starts:
        log(f"START {s_['sid']} ({s_['kind']}; {s_['origin']}; {s_['how']}) p ({', '.join(f'{v:.6f}' for v in s_['p'])}) "
            f"chi ({', '.join(f'{v:.6f}' for v in s_['chi'])})" + (f" grid S {s_['S0']!r} E[N] {s_['EN0']!r}" if 'S0' in s_ else ''))
    return starts

def nm_run(st):
    """one Nelder-Mead run, maximising F (minimising -F), resumable: evaluations already in the run's checkpoint file are replayed
    in order (the objective returns the archived value when the requested x equals the archived x bitwise)."""
    path = os.path.join(OUT, 'nm', f"eval_{st['sid']}.jsonl"); donep = os.path.join(OUT, 'nm', f"done_{st['sid']}.json")
    if os.path.exists(donep): return json.load(open(donep))
    cache = clean_jsonl(path); cnt = [0]
    x0 = np.asarray(st['x0'], float)
    def f(x):
        n = cnt[0]; cnt[0] += 1; xl = [float(v) for v in x]
        if n < len(cache):
            assert cache[n]['x'] == xl, (st['sid'], n)                # deterministic replay
            return -cache[n]['F']
        a, chi = x_to_input(x)
        rec = evaluate_u(a, chi, f"nm:{st['sid']}:{n+1}"); rec.update(stage='nm', sid=st['sid'], nfe=n+1, x=xl)
        append_jsonl(path, rec); return -rec['F']
    t0 = time.time()
    sim = eval(_sim_src.split('=', 1)[1].strip(), {'np': np, 'x0': x0, 'H_Z': H_Z, 'H_CHI': H_CHI})
    res = eval(_min_src.split('=', 1)[1].strip(), {'minimize': minimize, 'f': f, 'x0': x0, 'MAXFEV': MAXFEV, 'sim': sim})
    out = dict(sid=st['sid'], x=np.asarray(res.x).tolist(), negF=float(res.fun), nfev=int(res.nfev), nit=int(res.nit), msg=str(res.message),
               n_obj_calls=cnt[0], wall_s=time.time()-t0, replayed=min(len(cache), cnt[0]))
    json.dump(out, open(donep, 'w')); return out

def _nm_task(st): return nm_run(st)

def stage_nm(grid, log):
    os.makedirs(os.path.join(OUT, 'nm'), exist_ok=True)
    sp = os.path.join(OUT, 'nm_starts.json')
    if os.path.exists(sp): starts = json.load(open(sp)); log(f"NM starts read from {os.path.relpath(sp, HERE)} ({len(starts)})")
    else: starts = make_starts(grid, log); json.dump(starts, open(sp, 'w'), indent=1)
    log(f"NM {len(starts)} runs; protocol of stage2_local.py: {_line.strip()}; {_sim_src}; {_min_src}; objective -F, F = S - {MU} max(E[N] - B, 0)")
    t0 = time.time()
    with mp.get_context('fork').Pool(4, initializer=_init_worker, initargs=(JU,)) as pl:
        runs = []
        for r in pl.imap_unordered(_nm_task, starts, chunksize=1):
            runs.append(r); log(f"NM run {r['sid']} done: nfev {r['nfev']} nit {r['nit']} ({r['msg']}); -F at result {r['negF']!r}; "
                                f"replayed {r['replayed']}; wall {r['wall_s']:.0f} s; {len(runs)}/{len(starts)} runs, stage wall {time.time()-t0:.0f} s")
    byid = {r['sid']: r for r in runs}; runs = [byid[s_['sid']] for s_ in starts]
    recs = [r for s_ in starts for r in read_jsonl(os.path.join(OUT, 'nm', f"eval_{s_['sid']}.jsonl"))]
    log(f"NM evaluations {len(recs)} (sum of nfev {sum(r['nfev'] for r in runs)})")
    return starts, runs, recs


# --- stage: dense scan --------------------------------------------------------------------------------------------------------------
def dense_points(c):
    """coordinate-wise points around the centre record c (spec §4): phases chi_m +- 1..6 deg, powers p_m + t with the other three
    rescaled proportionally (direction skipped if p_m = 1; points with a negative power skipped)."""
    pts = []; p0 = np.array(c['p'], float); chi0 = np.array(c['chi'], float)
    for m in range(4):
        for d in PHASE_DEG:
            for sg in (+1, -1):
                chi = chi0.copy(); chi[m] = chi0[m]+sg*d*np.pi/180
                pts.append((np.sqrt(p0).tolist(), chi.tolist(), f'dense:chi{m+1}:{sg*d:+d}deg', dict(stage='dense', coord=f'chi{m+1}', step=f'{sg*d:+d} deg')))
    skipped = []
    for m in range(4):
        if p0[m] == 1.0: skipped.append(f'p{m+1}: direction skipped (p_m = 1)'); continue
        for t_ in POWER_T:
            for sg in (+1, -1):
                t = sg*t_; p = p0*(1.0-p0[m]-t)/(1.0-p0[m]); p[m] = p0[m]+t
                if (p < 0).any(): skipped.append(f'p{m+1} {t:+.3f}: negative power'); continue
                pts.append((np.sqrt(p).tolist(), chi0.tolist(), f'dense:p{m+1}:{t:+.3f}', dict(stage='dense', coord=f'p{m+1}', step=f'{t:+.3f}')))
    return pts, skipped

def stage_dense(centre, log):
    pts, skipped = dense_points(centre)
    log(f"DENSE centre {centre['tag']}: {fmt_rec(centre)}")
    log(f"DENSE {len(pts)} points ({sum(1 for p_ in pts if 'chi' in p_[3]['coord'])} phase, {sum(1 for p_ in pts if p_[3]['coord'][0] == 'p')} power); "
        f"skipped: {skipped if skipped else 'none'}")
    recs = run_pool(pts, os.path.join(OUT, 'eval_dense.jsonl'), log)
    for r in recs:
        log(f"DENSE {r['coord']} {r['step']}: S {r['S']!r} (S - S_centre {r['S']-centre['S']:+.6f}) E[N] {r['EN']!r} ({r['feas']}) F {r['F']!r}")
    return recs


# --- main ---------------------------------------------------------------------------------------------------------------------------
JU = None
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['all', 'grid', 'nm', 'dense', 'summary']); args = ap.parse_args()
    sys.stdout = dr.Tee(os.path.join(OUT, 'search_maxS.log'), 'a')
    T0 = time.time()
    def log(m): print(m, flush=True)
    log(f"\n=== search_maxS.py {args.stage} started {time.strftime('%Y-%m-%d %H:%M:%S')}; spec reports_CN/plan_round2b_CN.md v3.2 §4; "
        f"evaluator_fast.py (B, n = 24, margin 8 Delta); B = {B!r}, tau_N = {TAU_N} (round2a/config_recheck.json); mu = {MU}; cap {CAP}")
    cfg = dict(spec='reports_CN/plan_round2b_CN.md v3.2 §4', evaluator='evaluator_fast.py, version B, n = 24, margin 8 Delta',
               B=B, tau_N=TAU_N, source_B_tau='round2a/config_recheck.json', feasibility='round2a/recheck7.feasibility (reused)',
               guidance_objective=f'F(u) = S(u) - {MU} max(E[N](u) - B, 0) (search only; selection by the feasibility rule)',
               grid='6^4 phases chi_m in {0,60,...,300} deg at a_m = 0.5 (p_m = 1/4), itertools.product order of mc85_scan.py',
               nm=dict(protocol='stage2_pkg/stage2_local.py', parametrisation='x = (z1,z2,z3,chi1..chi4), p = softmax(z1,z2,z3,0), a = sqrt(p), chi mod 2 pi at evaluation',
                       h_z=H_Z, h_chi=H_CHI, maxfev=MAXFEV, xatol=1e-4, fatol=1e-5, method='Nelder-Mead', simplex=_sim_src, call=_min_src,
                       starts=f'{N_GRID_STARTS} grid points with highest S among 按数值判据可行 (top up by F) + 7 round-2a candidates + {N_RAND} random (seed {SEED_RAND}: rng.dirichlet(ones(4), 20) then rng.uniform(0, 2 pi, (20, 4)))'),
               dense=dict(phase_deg=PHASE_DEG, power_t=POWER_T, rule='coordinate-wise around S_found, centre fixed, no re-centring; power: others rescaled proportionally; skip p_m = 1 and negative powers'),
               cap=CAP, duplicates='not removed (不去重)')
    json.dump(cfg, open(os.path.join(OUT, 'config.json'), 'w'), indent=1, ensure_ascii=False)
    ef._warmup()
    recU, JU, rU = ef.evaluate_fast('U', 24)
    log(f"REF U E[N] {recU['EN']!r} equals frozen B bitwise: {recU['EN'] == B}; S {recU['S']!r}")
    assert recU['EN'] == B
    grid = stage_grid(log) if args.stage in ('all', 'grid', 'nm', 'dense', 'summary') else None
    if args.stage == 'grid': sys.exit(0)
    starts, runs, nmrecs = stage_nm(grid, log)
    if args.stage == 'nm': sys.exit(0)
    pre = grid+nmrecs
    feas = [r for r in pre if is_feasible(r)]
    centre = max(feas, key=lambda r: r['S'])
    log(f"S_found before the dense scan (largest S among 按数值判据可行, {len(feas)} of {len(pre)} archived): {centre['tag']} {fmt_rec(centre)}")
    dense = stage_dense(centre, log)
    allr = pre+dense
    json.dump(dict(B=B, tau_N=TAU_N, n=len(allr), records=allr), open(os.path.join(OUT, 'archive_all.json'), 'w'))
    feas_all = [r for r in allr if is_feasible(r)]
    sf = max(feas_all, key=lambda r: r['S']); champF = max(allr, key=lambda r: r['F'])
    dense_better = [r for r in dense if is_feasible(r) and r['S'] > centre['S']]
    log(f"DENSE feasible points with S above the centre: {len(dense_better)}" + (f"; best {max(dense_better, key=lambda r: r['S'])['tag']}" if dense_better else ''))
    log(f"S_FOUND {sf['S']!r} at {sf['tag']} (stage {sf['stage']}): {fmt_rec(sf)}; a ({', '.join(repr(v) for v in sf['a'])}); "
        f"chi ({', '.join(repr(v) for v in sf['chi'])})")
    log(f"F-champion (not used for selection): {champF['tag']} {fmt_rec(champF)}")
    cnt = {s_: sum(1 for r in allr if r['stage'] == s_) for s_ in ('grid', 'nm', 'dense')}
    log(f"COUNT evaluations: grid {cnt['grid']}, nm {cnt['nm']}, dense {cnt['dense']}, total {len(allr)} (cap {CAP}); "
        f"按数值判据可行 {len(feas_all)}, 边界未定 {sum(r['feas'] == '边界未定' for r in allr)}, 不可行 {sum(r['feas'] == '按数值判据不可行' for r in allr)}")
    rt = np.array([r['runtime_s'] for r in allr]); wt = np.array([r['wall_s'] for r in allr])
    log(f"TIME per evaluation (evaluator runtime_s, 4 concurrent workers): mean {rt.mean():.3f} s, median {np.median(rt):.3f} s, max {rt.max():.3f} s; "
        f"wall per evaluation incl. relative(): mean {wt.mean():.3f} s; sum of evaluation walls {wt.sum()/3600:.3f} h")
    json.dump(dict(S_found=sf, centre_before_dense=centre['tag'], dense_improved=bool(dense_better), F_champion=champF['tag'],
                   counts=dict(cnt, total=len(allr), cap=CAP)), open(os.path.join(OUT, 'S_found.json'), 'w'), indent=1, ensure_ascii=False)
    log(f"=== search_maxS.py {args.stage} finished {time.strftime('%Y-%m-%d %H:%M:%S')}; this invocation wall {time.time()-T0:.0f} s")
