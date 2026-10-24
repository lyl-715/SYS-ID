"""Round 5, Part A: the frozen round-4 threshold tables evaluated under model mismatch (spec reports_CN/plan_round5_CN.md v3.2,
frozen; §1.1, §1.2, §2 items 1-3). 冻结表在失配模型下的离散数值评价 -- not a bound, not a robust design.

True system theta: x_k(theta) = h_k^T theta, built as x(theta) = xbar + H (theta - theta_0) from the nominal output array xbar
exactly as the round-4 implementation produced it (dr.input_arrays; never H theta_0 recomputed in another floating-point
order), so theta = theta_0 reproduces round 4 bit for bit. theta_0 = [G(0), Re G(j w_1), Im G(j w_1), ..., Re G(j w_4),
Im G(j w_4)] (the regressor layout of mc83.py: h_k = [a0, a_m cos psi_m, -a_m sin psi_m]).

Frozen (never rebuilt from the perturbed output): the nominal reference grid z_i = z_0 + i h, h = sigma/24 (dr.make_grid of the
nominal xbar); every step's table start index and width (round4/results/table_<case>.npz); the physical lookup policy (node i
looks up the nearest stored node -- itself on this grid -- and a node outside the step's stored range takes the boundary node's
action; physical half-widths M^+- h; window centred at the node's own r). The perturbed mean x_k(theta) is used everywhere the
model enters: the kernel ghat_{k,j}(theta) ~ phi((r_j - x_k(theta))/sigma) on the step's support window, the initial
distribution m_2 = ghat_1(theta), alpha and beta. The support windows (a computational device that must hold the mass) are
the round-4 formula centred at x_k(theta) on the frozen grid (DECISION A-2). sigma, h_k (hence J's assembly) and the table
are unchanged; J(theta) = sigma^-2 sum_k w_k(theta) h_k h_k^T at the true theta.

Constant-Delta SoD control at the same theta: evaluator_fast.evaluate_fast (accepted operator, R = 68 sigma) with its input
arrays replaced by (x(theta), H) and its grid pinned to the nominal grid (DECISION A-3); at theta_0 this is evaluator_fast
unchanged. Tables: the round-4 general window path (evaluator_window.forward, R = 88 sigma) with record and J from the accepted
dr.evaluate.

Perturbations (spec §1.2), eps := max_k |x_k(theta) - xbar_k|/sigma, normalised separately for each input:
  P1  theta_1 += sign eps sigma / max_k |h_k^T e_1|   (= eps sigma, h_k^T e_1 = a0 = 1)      eps in {0.1, 0.2, 0.5, 1, 2, 4}, +-
  P2  theta_{2..9} *= 1 + eta, eta = sign eps sigma / max_k |h_k^T (0, theta_0[2..9])|         eps in {0.1, 0.2, 0.5, 1, 2}, +-
  P3  theta_i += sign 0.5 sigma / max_k |h_k^T e_i|, i = 1..9                                    18 points
  P4  theta_0 + eps sigma d_j / max_k |h_k^T d_j|, d_j = row j of default_rng(20261025).standard_normal((10, 9)), normalised
      to unit length (shared raw directions; the same ten at every eps)                          eps in {0.2, 0.5, 1, 2}
Pairing: U/S and U/W at U's perturbed theta with U's constant-Delta SoD at that theta; P5514/W at P5514's own perturbed theta
with P5514's SoD at that theta. 80 points per input; 3 + 2 evaluations per point pair (400), plus the theta_0 regression (all
four tables, both SoDs).

Usage (from this directory):
  python3 robustness.py check     # spec §2 items 1-3 at theta_0 -> logs/partA_check.log, results/partA_check.json
  python3 robustness.py sweep     # 400 evaluations, 4 workers -> results/archive_partA.jsonl, results/J_partA.npz, logs/partA_sweep.log
  python3 robustness.py figs      # fig_mismatch_dmD.png, fig_mismatch_EN.png
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, argparse
from contextlib import contextmanager
HERE = os.path.dirname(os.path.abspath(__file__)); R4 = os.path.join(os.path.dirname(HERE), 'round4')
sys.path.insert(0, R4)
import numpy as np
import evaluator_window as ew
import evaluator_fast as ef
dr = ew.dr

s, D, K, B = ew.s, ew.D, ew.K, ew.B_BUDGET
N_GRID = ew.N_GRID; X = ew.X24
LOGS = os.path.join(HERE, 'logs'); RES = os.path.join(HERE, 'results')
os.makedirs(LOGS, exist_ok=True); os.makedirs(RES, exist_ok=True)
ARCHIVE = os.path.join(RES, 'archive_partA.jsonl'); JNPZ = os.path.join(RES, 'J_partA.npz')
VERDICT = '离散方案下'

THETA0 = np.concatenate([[dr.G(0)], np.column_stack([dr.Gj.real, dr.Gj.imag]).ravel()])
EPS_P1 = (0.1, 0.2, 0.5, 1.0, 2.0, 4.0); EPS_P2 = (0.1, 0.2, 0.5, 1.0, 2.0); EPS_P3 = 0.5; EPS_P4 = (0.2, 0.5, 1.0, 2.0)
P4_SEED, N_DIR = 20261025, 10
SWEEP_TABLES = {'U': ('S', 'W'), 'P5514': ('W',)}          # spec §1.2: three tables, two controls per point
ALL_CASES = ('U/S', 'U/W', 'P5514/S', 'P5514/W')            # spec §2 item 1: all four at theta_0


# --- inputs, frozen grid, tables -----------------------------------------------------------------------------------------
INP = {}
for _n in ew.INPUTS:
    _x, _H = dr.input_arrays(_n); _z, _dz = dr.make_grid(_x, N_GRID, 8*D, 'B')
    INP[_n] = dict(xbar=_x, H=_H, z=_z, dz=_dz, G=len(_z))

class LookupTable:
    """a frozen round-4 table on the frozen sigma/24 grid with the runtime lookup rule: node i of step k (array index p) takes
    the stored action of node clip(i, start, start + width - 1) (the nearest stored node is i itself on the same grid; outside
    the step's stored range, the boundary node). At theta_0 the support windows equal the stored ranges and rows() equals
    evaluator_window.CodeTable.rows element for element."""
    def __init__(self, codes, start, width):
        self.codes, self.start, self.width = codes, np.asarray(start, np.int64), np.asarray(width, np.int64); self.X = X
    def rows(self, p, a, b):
        r = p-1; s0 = int(self.start[r]); w = int(self.width[r])
        o = np.clip(np.arange(a, b+1), s0, s0+w-1)-s0
        return ew.code_to_m(self.codes[r, o])
    def is_sod(self): return False

def load_tables(cases=ALL_CASES):
    """round4/results/table_<case>.npz, checked against the nominal design grid's support windows (as part_c.load_table)."""
    T = {}
    for case in cases:
        name = case.split('/')[0]; I = INP[name]
        a, b = ew.windows88(I['xbar'], I['z'][0], I['dz'], I['G'])
        with np.load(os.path.join(R4, 'results', f"table_{case.replace('/', '_')}.npz")) as f:
            codes, start, width, lam, run = f['codes'], f['start'], f['width'], float(f['lam']), int(f['run'])
        assert np.array_equal(start, a[1:]) and np.array_equal(width, (b-a+1)[1:]), case
        T[case] = LookupTable(codes, start, width); T[case].lam, T[case].run = lam, run
    return T


# --- perturbations --------------------------------------------------------------------------------------------------------
def p4_directions():
    Z = np.random.default_rng(P4_SEED).standard_normal((N_DIR, 9))
    return Z/np.linalg.norm(Z, axis=1, keepdims=True)

def points(name):
    """the 80 perturbation points of spec §1.2 for one input (normalised with that input's h_k)."""
    H = INP[name]['H']; pts = []
    def add(cls, eps_t, sign, u, tag):
        nrm = float(np.max(np.abs(H@u))); step = sign*eps_t*s/nrm
        th = THETA0+step*u
        pts.append(dict(cls=cls, eps_target=eps_t, sign=sign, dir=tag, norm=nrm, step=step, theta=th))
    e1 = np.eye(9)
    for e in EPS_P1:
        for sg in (1, -1): add('P1', e, sg, e1[0], 'theta1')
    u2 = np.r_[0.0, THETA0[1:]]
    for e in EPS_P2:
        for sg in (1, -1): add('P2', e, sg, u2, 'theta2..9 x (1+eta)')     # step = eta
    for i in range(9):
        for sg in (1, -1): add('P3', EPS_P3, sg, e1[i], f'theta{i+1}')
    dirs = p4_directions()
    for e in EPS_P4:
        for j in range(N_DIR): add('P4', e, 1, dirs[j], f'd{j+1}')
    assert len(pts) == 80
    return pts

def x_of(name, theta):
    """x(theta) = xbar + H (theta - theta_0) (spec §1.1; at theta_0 the nominal array itself, bit for bit)."""
    I = INP[name]; return I['xbar']+I['H']@(theta-THETA0)


# --- evaluation at a true theta ------------------------------------------------------------------------------------------
@contextmanager
def patched(mod, **kw):
    old = {k: getattr(mod, k) for k in kw}
    for k, v in kw.items(): setattr(mod, k, v)
    try: yield
    finally:
        for k, v in old.items(): setattr(mod, k, v)

def eval_table(name, tab, xth):
    """the frozen table at the true output xth: round-4 general path on the frozen grid, support windows centred at xth;
    record and J from the accepted dr.evaluate (its input arrays replaced by (xth, H) for this call only)."""
    I = INP[name]; z, dz, G = I['z'], I['dz'], I['G']
    a, b = ew.windows88(xth, z[0], dz, G)
    assert int((a-tab.X).min()) >= 0 and int((b+tab.X).max()) <= G-1, "a window would cross the frozen grid"
    def run(xk, n_, margin, version):
        assert xk is xth and n_ == N_GRID and version == 'B'
        return ew.forward(z, dz, s, xth, a, b, tab)
    with patched(dr, input_arrays=lambda nm: (xth, I['H']), run_ab=run):
        rec, J, r = dr.evaluate('B', name, N_GRID, 8)
    rec['path'] = 'general window path (R = 88 sigma), frozen grid, windows centred at x(theta)'
    # boundary lookups actually used: steps whose mass support (m_k at step p = sup of index p - 1) leaves the stored range
    lo, hi = r['sup_lo'][:-1], r['sup_hi'][:-1]; st, en = tab.start, tab.start+tab.width-1
    out_lo = np.maximum(st-lo, 0); out_hi = np.maximum(hi-en, 0)
    rec['boundary_steps'] = int(((out_lo > 0) | (out_hi > 0)).sum()); rec['boundary_nodes_max'] = int(max(out_lo.max(), out_hi.max()))
    return rec, J, r

def eval_sod(name, xth):
    """constant-Delta SoD at the true output xth: evaluator_fast.evaluate_fast with input arrays (xth, H) and its grid pinned to
    the frozen nominal grid (dr.make_grid called with the nominal xbar); at theta_0 nothing is replaced in effect."""
    I = INP[name]; mg = dr.make_grid
    with patched(dr, input_arrays=lambda nm: (xth, I['H']), make_grid=lambda xk, n, margin, version: mg(I['xbar'], n, margin, version)):
        rec, J, r = ef.evaluate_fast(name, N_GRID)
    rec['path'] = 'accepted operator (evaluator_fast.evaluate_fast), frozen grid, windows centred at x(theta)'
    rec['assert_fail'] = int((~r['assert_ok']).sum())
    return rec, J, r

KEEP = ('EN', 'EN_P', 'S', 'logdet', 'logdet_sign', 'mass_dev_max', 'min_m', 'kernel_mass_dev_max', 'w_min', 'w_max',
        'inv_S', 'inv_J', 'inv_w', 'runtime_s', 'path', 'boundary_steps', 'boundary_nodes_max')

def evaluate_point(name, pt, tables, which):
    """SoD and the listed tables of one input at one point; one record per evaluation (assertion stops recorded, not raised)."""
    th = pt['theta']; xth = x_of(name, th); d = xth-INP[name]['xbar']
    eps_att = float(np.max(np.abs(d))/s); disp = float(np.max(np.abs(np.diff(xth)))/s)
    base = dict(input=name, cls=pt['cls'], eps_target=pt['eps_target'], sign=pt['sign'], dir=pt['dir'], norm=pt['norm'],
                step=pt['step'], theta=[float(v) for v in th], eps_attained=eps_att, max_disp_sigma=disp)
    out = []
    for tname in ('SoD',)+tuple(which):
        rec = dict(base, table=tname); t0 = time.perf_counter()
        try:
            if tname == 'SoD': r_, J, rr = eval_sod(name, xth)
            else: r_, J, rr = eval_table(name, tables[f'{name}/{tname}'], xth)
            rec.update({k: (r_[k].item() if hasattr(r_[k], 'item') else r_[k]) for k in KEEP if k in r_})
            rec['assert_fail'] = int((~rr['assert_ok']).sum()); rec['status'] = 'ok'
            rec['J'] = J
        except (ew.AssertionStop, ef.AssertionStop) as e:
            rec.update(status='assert_stop', message=str(e))
        rec['wall_s'] = time.perf_counter()-t0
        out.append(rec)
    return out


# --- check: theta_0 regression (spec §2 items 1-3) --------------------------------------------------------------------------
def check(out):
    ew._warmup()
    T = load_tables(); c1 = json.load(open(os.path.join(R4, 'results', 'c1.json'))); res = {}; ok_all = True
    for name in ew.INPUTS:
        x0 = x_of(name, THETA0.copy())
        out(f"A-CHK {name} x(theta_0) bitwise equal to the nominal array: {np.array_equal(x0, INP[name]['xbar'])}; "
            f"max |H theta_0 - xbar| / sigma (diagnostic, not used) {float(np.max(np.abs(INP[name]['H']@THETA0-INP[name]['xbar']))/s)!r}")
        # constant-Delta SoD vs evaluator_fast unchanged
        rs, Js, rrs = eval_sod(name, x0); r0, J0, rr0 = ef.evaluate_fast(name, N_GRID)
        same = (rs['EN'] == r0['EN'] and rs['S'] == r0['S'] and rs['logdet'] == r0['logdet'] and rs['EN_P'] == r0['EN_P']
                and np.array_equal(Js, J0) and np.array_equal(rrs['W'], rr0['W']))
        res[f'{name}/SoD'] = dict(bitwise_vs_evaluator_fast=same, EN=rs['EN'], S=rs['S'], logdet=rs['logdet'], c1_EN=c1[f'{name}/SoD']['EN'],
                                  c1_equal=rs['EN'] == c1[f'{name}/SoD']['EN'] and rs['logdet'] == c1[f'{name}/SoD']['logdet'])
        ok_all &= same and res[f'{name}/SoD']['c1_equal']
        out(f"A-CHK {name}/SoD EN {rs['EN']!r} S {rs['S']!r} logdet {rs['logdet']!r}; bitwise vs evaluator_fast (E[N], E[N]_P, S, "
            f"log det, 81 J, 5000 w_k): {same}; equal to round-4 c1.json: {res[f'{name}/SoD']['c1_equal']}")
        Jsod = Js
        for case in [c for c in ALL_CASES if c.startswith(name+'/')]:
            rt, Jt, rrt = eval_table(name, T[case], x0)
            ref = c1[case]; Jref = np.load(os.path.join(R4, 'results', f"J_{case.replace('/', '_')}.npy"))
            rw, Jw, rrw = ew.evaluate_table(name, ew.CodeTable(T[case].codes, T[case].start, T[case].width))   # round-4 path, in-process
            eqc1 = all(rt[k] == ref[k] for k in ('EN', 'EN_P', 'S', 'logdet'))
            eqJ = bool(np.array_equal(Jt, Jref)); eqW = bool(np.array_equal(rrt['W'], rrw['W']) and np.array_equal(rrt['Q'], rrw['Q']))
            dmD = float(rt['logdet']-np.linalg.slogdet(Jsod)[1])
            res[case] = dict(EN=rt['EN'], S=rt['S'], logdet=rt['logdet'], dmD=dmD, c1_dmD=ref['dmD'], equal_c1=eqc1, J_bitwise=eqJ,
                             w_q_bitwise_vs_round4_path=eqW, dmD_equal=dmD == ref['dmD'], boundary_steps=rt['boundary_steps'],
                             mass_dev_max=rt['mass_dev_max'], min_m=rt['min_m'], assert_fail=int((~rrt['assert_ok']).sum()))
            ok = eqc1 and eqJ and eqW and dmD == ref['dmD']; ok_all &= ok
            for k in ('EN', 'S', 'logdet', 'dmD'):
                out(f"A-CHK {case} {k} {res[case][k]!r} (round 4 c1.json {ref[k]!r}) equal {res[case][k] == ref[k]}")
            out(f"A-CHK {case} J (81 elements) bitwise equal to round4/results/J_{case.replace('/', '_')}.npy: {eqJ}; w_k and E[q_k] bitwise "
                f"equal to the round-4 path (evaluator_window.evaluate_table, CodeTable) run now: {eqW}; boundary lookups with mass "
                f"{rt['boundary_steps']}; mass dev {rt['mass_dev_max']!r}, min m {rt['min_m']!r}, assertion failures {res[case]['assert_fail']}")
    out(f"A-CHK spec §2 items 1-2 (theta_0 regression, four tables and two SoDs, bit for bit): {'PASS' if ok_all else 'FAIL'}")
    json.dump(dict(results=res, pass_=ok_all), open(os.path.join(RES, 'partA_check.json'), 'w'), indent=1)
    return ok_all


# --- sweep ------------------------------------------------------------------------------------------------------------------
_T = None
def _init():
    global _T
    ew._warmup()

def _task(args):
    name, k = args
    pt = POINTS[name][k]
    return name, k, evaluate_point(name, pt, _T, SWEEP_TABLES[name])

POINTS = {n: points(n) for n in ew.INPUTS}

def sweep(out, workers=4):
    global _T
    import multiprocessing as mp
    _T = load_tables(('U/S', 'U/W', 'P5514/W'))
    for name in ew.INPUTS:
        nrm = {}
        for p in POINTS[name]: nrm.setdefault((p['cls'], p['dir']), p['norm'])
        for (c, d_), v in nrm.items(): out(f"A-NORM {name} {c} {d_} max_k |h_k^T u| {v!r}")
    for j, d in enumerate(p4_directions()): out(f"A-DIR d{j+1} {[float(v) for v in d]!r}")
    jobs = [(n, k) for n in ew.INPUTS for k in range(len(POINTS[n]))]
    t0 = time.perf_counter()
    with mp.get_context('fork').Pool(workers, initializer=_init) as pool:
        res = pool.map(_task, jobs, chunksize=1)
    wall = time.perf_counter()-t0
    # pairing with the same input's SoD at the same theta; nominal Delta m_D from the theta_0 check
    chk = json.load(open(os.path.join(RES, 'partA_check.json')))['results']
    lines = []; Js = {}
    for name, k, recs in res:
        sod = recs[0]; assert sod['table'] == 'SoD'
        for r in recs:
            r['id'] = f"{name}|{r['table']}|{r['cls']}|{r['dir']}|{'+' if r['sign'] > 0 else '-'}|{r['eps_target']}"
            if r['status'] == 'ok':
                Js[r['id']] = r.pop('J')
                r['budget_ok'] = bool(r['EN'] <= B); r['budget_label'] = VERDICT; r['slack'] = B-r['EN']
                if r['table'] == 'SoD':
                    r['dmD'] = 0.0; r['dlogdet_vs_nominal'] = r['logdet']-chk[f'{name}/SoD']['logdet']
                elif sod['status'] == 'ok':
                    r['logdet_SoD'] = sod['logdet']; r['EN_SoD'] = sod['EN']; r['dmD'] = r['logdet']-sod['logdet']
                    r['dmD_nominal'] = chk[f"{name}/{r['table']}"]['dmD']; r['retention'] = r['dmD']/r['dmD_nominal']
                else:
                    r['dmD'] = None
            else:
                r.pop('J', None)
            lines.append(r)
    with open(ARCHIVE, 'w') as f:
        for r in lines: f.write(json.dumps(r, ensure_ascii=False)+'\n')
    np.savez(JNPZ, **{k: v for k, v in Js.items()})
    # text table, one line per evaluation (same order as the archive)
    hdr = (f"{'input':6s} {'table':5s} {'cls':3s} {'dir':20s} {'sg':2s} {'eps_t':>5s} {'eps_att':>12s} {'E[N]':>10s} {'S':>10s} "
           f"{'logdet':>11s} {'dmD':>9s} {'keep':>7s} {'E[N]<=B':>8s} {'bnd':>4s}")
    out(f"A-TAB {hdr}")
    for r in lines:
        if r['status'] != 'ok':
            out(f"A-TAB {r['input']:6s} {r['table']:5s} {r['cls']:3s} {r['dir']:20s} {'+' if r['sign'] > 0 else '-':2s} {r['eps_target']:5.2f} STOP {r['message']}")
            continue
        out(f"A-TAB {r['input']:6s} {r['table']:5s} {r['cls']:3s} {r['dir']:20s} {'+' if r['sign'] > 0 else '-':2s} {r['eps_target']:5.2f} "
            f"{r['eps_attained']:12.9f} {r['EN']:10.4f} {r['S']:10.3f} {r['logdet']:11.5f} "
            f"{(r['dmD'] if r['dmD'] is not None else float('nan')):+9.4f} {r.get('retention', float('nan')):7.3f} "
            f"{('yes' if r['budget_ok'] else 'NO'):>8s} {r.get('boundary_steps', 0):4d}")
    ok = [r for r in lines if r['status'] == 'ok']
    nstop = len(lines)-len(ok)
    out(f"A-SUM evaluations {len(lines)} (ok {len(ok)}, assertion stops {nstop}); wall {wall:.1f} s with {workers} workers; "
        f"sum of per-evaluation wall {sum(r['wall_s'] for r in lines):.1f} s")
    out(f"A-SUM max |sum m - 1| {max(r['mass_dev_max'] for r in ok)!r}; min m {min(r['min_m'] for r in ok)!r}; assertion failures "
        f"{sum(r['assert_fail'] for r in ok)}; invariants S, J, w all true: {all(r['inv_S'] and r['inv_J'] and r['inv_w'] for r in ok)}")
    out(f"A-SUM max |eps_attained - eps_target| / eps_target {max(abs(r['eps_attained']-r['eps_target'])/r['eps_target'] for r in lines)!r}; "
        f"max |x_(k+1)(theta) - x_k(theta)| / sigma {max(r['max_disp_sigma'] for r in lines)!r} (support invariant uses 5)")
    out(f"A-SUM table evaluations using a boundary lookup with mass: {sum(1 for r in ok if r['table'] != 'SoD' and r['boundary_steps'] > 0)}")
    json.dump(dict(wall_s=wall, workers=workers, n=len(lines), n_ok=len(ok), n_stop=nstop), open(os.path.join(RES, 'partA_sweep.json'), 'w'), indent=1)
    return lines


# --- figures ------------------------------------------------------------------------------------------------------------------
def read_archive():
    return [json.loads(l) for l in open(ARCHIVE)]

def figs(out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    A = [r for r in read_archive() if r['status'] == 'ok']
    chk = json.load(open(os.path.join(RES, 'partA_check.json')))['results']
    INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    C = {'P1': '#2a78d6', 'P2': '#eb6834', 'P3': '#1baf7a', 'P4': '#6250d6'}
    plt.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                         'axes.facecolor': SURF, 'figure.facecolor': SURF, 'font.size': 8.5, 'axes.titlesize': 9})
    TABS = (('U', 'S', 'U / S table'), ('U', 'W', 'U / weighted table'), ('P5514', 'W', 'P5514 / weighted table'))
    for qty, fname, ylab in (('dmD', 'fig_mismatch_dmD.png', r'$\Delta m_D$ vs paired constant-$\Delta$ SoD (nat)'),
                             ('EN', 'fig_mismatch_EN.png', r'$\mathbb{E}[N]$ (discrete evaluator)')):
        fig, axs = plt.subplots(3, 3, figsize=(13, 10), sharey='row', constrained_layout=True)
        for i, (name, tb, lab) in enumerate(TABS):
            rows = [r for r in A if r['input'] == name and r['table'] == tb]
            nom = chk[f'{name}/{tb}'][qty if qty == 'EN' else 'dmD']
            ax = axs[i, 0]
            for cls in ('P1', 'P2'):
                rr = sorted([r for r in rows if r['cls'] == cls], key=lambda r: r['sign']*r['eps_attained'])
                xs = [0.0]+[r['sign']*r['eps_attained'] for r in rr]; ys = [nom]+[r[qty] for r in rr]
                o = np.argsort(xs); ax.plot(np.array(xs)[o], np.array(ys)[o], '-o', color=C[cls], lw=2, ms=4.5,
                                             label=('P1  theta_1 + eps sigma' if cls == 'P1' else 'P2  theta_2..9 x (1 + eta)'))
            ax.set_xscale('symlog', linthresh=0.1); ax.set_xlabel(r'signed $\varepsilon$ (attained)')
            tk = [-4, -2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2, 4]
            ax.set_xticks(tk); ax.set_xticklabels([f'{v:g}' for v in tk]); ax.minorticks_off()
            ax.set_title(f'{lab}: P1, P2', loc='left', color=INK)
            ax = axs[i, 1]
            for sg, mk in ((1, 'o'), (-1, 's')):
                rr = sorted([r for r in rows if r['cls'] == 'P3' and r['sign'] == sg], key=lambda r: int(r['dir'][5:]))
                ax.plot([int(r['dir'][5:])+(0.12*sg) for r in rr], [r[qty] for r in rr], mk, color=C['P3'], ms=6,
                        mfc=(C['P3'] if sg > 0 else SURF), mew=1.6, ls='none', label=f"P3 {'+' if sg > 0 else '-'}0.5 (theta_i alone)")
            ax.set_xticks(range(1, 10)); ax.set_xlabel(r'perturbed parameter $i$ ($\varepsilon=0.5$)')
            ax.set_title(f'{lab}: P3', loc='left', color=INK)
            ax = axs[i, 2]
            for j in range(N_DIR):
                rr = sorted([r for r in rows if r['cls'] == 'P4' and r['dir'] == f'd{j+1}'], key=lambda r: r['eps_target'])
                ax.plot([0.0]+[r['eps_target'] for r in rr], [nom]+[r[qty] for r in rr], '-', color=C['P4'], lw=0.9, alpha=0.55,
                        label=('P4 ten directions' if j == 0 else None))
            med = [nom]+[float(np.median([r[qty] for r in rows if r['cls'] == 'P4' and r['eps_target'] == e])) for e in EPS_P4]
            ax.plot([0.0]+list(EPS_P4), med, '-o', color=INK, lw=2.2, ms=5, label='median over directions')
            ax.set_xlabel(r'$\varepsilon$'); ax.set_xticks([0, 0.2, 0.5, 1, 2])
            ax.set_title(f'{lab}: P4', loc='left', color=INK)
            for ax in axs[i]:
                ax.axhline(nom, color=INK2, lw=0.9, ls=':', zorder=0)
                if qty == 'dmD':
                    ax.axhline(nom/2, color=INK2, lw=0.9, ls='--', zorder=0); ax.axhline(0.0, color=INK2, lw=0.6, zorder=0)
                else:
                    ax.axhline(B, color='#e34948', lw=1.0, ls='--', zorder=0)
                ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
                for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
            axs[i, 0].set_ylabel(ylab)
            if i == 0:
                for ax in axs[0]: ax.legend(frameon=False, fontsize=7.5, loc='best', labelcolor=INK)
        ref = ('dotted: nominal value at theta_0; dashed: half of it; solid: 0' if qty == 'dmD'
               else f'dotted: nominal E[N] at theta_0; dashed red: budget B = {B:.4f}')
        fig.suptitle(f"Frozen round-4 tables under model mismatch ({'Delta m_D' if qty == 'dmD' else 'E[N]'}); "
                     f"discrete evaluation, sigma/24 grid. {ref}", color=INK, fontsize=10, x=0.01, ha='left')
        fig.savefig(os.path.join(HERE, fname), dpi=130); plt.close(fig)
        out(f"A-FIG {fname} written")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['check', 'sweep', 'figs'])
    ap.add_argument('--workers', type=int, default=4)
    A_ = ap.parse_args()
    sys.stdout = dr.Tee(os.path.join(LOGS, f'partA_{A_.stage}.log'))
    import numba, scipy
    print(f"=== robustness.py {A_.stage} started {time.strftime('%Y-%m-%d %H:%M:%S')}; spec reports_CN/plan_round5_CN.md v3.2; python "
          f"{sys.version.split()[0]}, numpy {np.__version__}, scipy {scipy.__version__}, numba {numba.__version__}; B = {B!r}; "
          f"theta_0 = {[float(v) for v in THETA0]!r}")
    t0 = time.perf_counter()
    if A_.stage == 'check': ok = check(print)
    elif A_.stage == 'sweep': sweep(print, A_.workers)
    else: figs(print)
    print(f"=== robustness.py {A_.stage} done in {time.perf_counter()-t0:.1f} s")
