"""Round 6, Parts A2-A5: the four uniform-input weighted tables (m = 0, 0.05, 0.10, 0.20) at round 5's 80 perturbation points
(spec reports_CN/plan_round6_CN.md v3.2, frozen; §1.2, §1.3, §4 item 1). 固定设计、离散方案下 -- discrete evaluation of fixed
tables under model mismatch, not a bound and not a robust design.

The evaluation is round 5's code itself, imported unchanged (round5/robustness.py: points('U'), x_of, LookupTable, eval_table,
eval_sod, evaluate_point; frozen sigma/24 grid and absolute coordinates, per-input normalisation, pairing with the constant-Delta
SoD at the same theta). The tables enter its table dictionary under the names m000 (round4/results/table_U_W.npz) and
m005, m010, m020 (results/table_U_W_m*.npz of margin_design.py). theta_0 is evaluated through the same path.

  sweep    81 points (theta_0 + 80) x (SoD + four tables) = 405 evaluations, 4 workers -> results/archive_partA.jsonl (one line per
           evaluation), results/J_partA.npz; regression against round 5 (theta_0: check 1; all 80 points: m000 and SoD)
  summary  A3-A5 from the archive: 26-point budget pass, m*, rho_26, per-table rows, g_med -> results/partA_summary.json
  figs     fig_margin_EN_dmD.png (E[N] and Delta m_D vs eps, coloured by m), fig_margin_cost_slack.png (nominal gain and minimum
           slack vs m)

Usage (from this directory): python3 robustness_margin.py sweep [--outdir DIR] [--max-new N] | summary | figs

Round 7 Part 0 (no numerical change): `sweep` writes a per-point incremental checkpoint, results/partA_sweep.ckpt.jsonl (one
line per completed point: its five evaluation records with the 81 J elements as exact JSON floats, appended and fsync'ed as
the point completes; not committed); a re-run resumes from it and evaluates only the missing points. The archive and
J_partA.npz are assembled from the checkpoint in point order once all 81 points are done (same format and order as before).
--outdir redirects every output of `sweep` (archive, J npz, checkpoint, partA_sweep.json, log) to DIR (verification runs);
--max-new N stops after N newly completed points (an interruption, to exercise the resume).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
R4 = os.path.join(EV, 'round4'); R5 = os.path.join(EV, 'round5')
sys.path.insert(0, R4); sys.path.insert(0, R5)
import numpy as np
import robustness as rb                                   # round 5, unchanged
ew, ef, dr = rb.ew, rb.ef, rb.dr

s, K, B = rb.s, rb.K, rb.B
assert B == 316.8921168493721
LOGS = os.path.join(HERE, 'logs'); RES = os.path.join(HERE, 'results')
os.makedirs(LOGS, exist_ok=True); os.makedirs(RES, exist_ok=True)
ARCHIVE = os.path.join(RES, 'archive_partA.jsonl'); JNPZ = os.path.join(RES, 'J_partA.npz'); SUMMARY = os.path.join(RES, 'partA_summary.json')
OUT = dict(archive=ARCHIVE, jnpz=JNPZ, ckpt=os.path.join(RES, 'partA_sweep.ckpt.jsonl'), sweep=os.path.join(RES, 'partA_sweep.json'))

def set_outdir(d):
    """redirect the outputs of `sweep` to directory d (verification runs); inputs are unchanged."""
    os.makedirs(d, exist_ok=True)
    OUT.update(archive=os.path.join(d, 'archive_partA.jsonl'), jnpz=os.path.join(d, 'J_partA.npz'), ckpt=os.path.join(d, 'partA_sweep.ckpt.jsonl'),
               sweep=os.path.join(d, 'partA_sweep.json'))
VERDICT = '离散方案下'
NAME = 'U'
TABS = ('m000', 'm005', 'm010', 'm020')
M_OF = {'m000': 0.0, 'm005': 0.05, 'm010': 0.10, 'm020': 0.20}
TABLE_NPZ = {'m000': os.path.join(R4, 'results', 'table_U_W.npz'), **{t: os.path.join(RES, f'table_U_W_{t}.npz') for t in TABS[1:]}}
EPS_P4 = rb.EPS_P4


def in_selection(r):
    """the frozen 26-point selection set (spec §1.3): P4 eps in {0.2, 0.5} (ten directions) + P1 eps in {0.1, 0.2, 0.5}, +-."""
    return (r['cls'] == 'P4' and r['eps_target'] in (0.2, 0.5)) or (r['cls'] == 'P1' and r['eps_target'] in (0.1, 0.2, 0.5))

def point_id(r): return f"{r['cls']}|{r['dir']}|{'+' if r['sign'] > 0 else '-'}|{r['eps_target']}"

def load_tables():
    """the four tables as round-5 LookupTables, each checked against the nominal design grid's support windows (as
    rb.load_tables) and, for the margins, against margin_design's final record (run, lambda)."""
    I = rb.INP[NAME]; a, b = ew.windows88(I['xbar'], I['z'][0], I['dz'], I['G'])
    fin = json.load(open(os.path.join(RES, 'margin_final.json')))
    T = {}
    for t in TABS:
        with np.load(TABLE_NPZ[t]) as f:
            codes, start, width, lam, run = f['codes'], f['start'], f['width'], float(f['lam']), int(f['run'])
        assert np.array_equal(start, a[1:]) and np.array_equal(width, (b-a+1)[1:]), t
        assert run == fin[t]['run'] and lam == fin[t]['lam'], (t, run, fin[t]['run'], lam, fin[t]['lam'])
        T[f'{NAME}/{t}'] = rb.LookupTable(codes, start, width); T[f'{NAME}/{t}'].lam, T[f'{NAME}/{t}'].run = lam, run
    return T


# --- sweep --------------------------------------------------------------------------------------------------------------------
_T = None
def _init(): ew._warmup()

def _points():
    p0 = dict(cls='theta0', eps_target=0.0, sign=1, dir='theta_0', norm=float('nan'), step=0.0, theta=rb.THETA0.copy())
    return [p0]+rb.POINTS[NAME]

def _task(k):
    return k, rb.evaluate_point(NAME, _points()[k], _T, TABS)

def sweep(out, workers=4, max_new=None):
    global _T
    import multiprocessing as mp
    _T = load_tables()
    for t in TABS: out(f"A2-TABLE {t} {os.path.relpath(TABLE_NPZ[t], HERE)} run {_T[f'{NAME}/{t}'].run} lam {_T[f'{NAME}/{t}'].lam!r}")
    pts = _points(); assert len(pts) == 81
    # per-point incremental checkpoint (round 7 Part 0): one line per completed point, J as exact JSON floats
    done = {}
    if os.path.exists(OUT['ckpt']):
        for l in open(OUT['ckpt']):
            c = json.loads(l); done[c['k']] = c
    todo = [k for k in range(len(pts)) if k not in done]
    out(f"A2-CKPT {OUT['ckpt']}: {len(done)} points resumed, {len(todo)} to evaluate")
    t0 = time.perf_counter(); n_new = 0
    if todo:
        with mp.get_context('fork').Pool(workers, initializer=_init) as pool, open(OUT['ckpt'], 'a') as fc:
            for k, recs in pool.imap_unordered(_task, todo, chunksize=1):
                c = dict(k=k, recs=[{**r, **({'J': r['J'].tolist()} if 'J' in r else {})} for r in recs])
                fc.write(json.dumps(c, ensure_ascii=False)+'\n'); fc.flush(); os.fsync(fc.fileno())
                done[k] = c; n_new += 1
                if max_new is not None and n_new >= max_new:
                    pool.terminate(); break
    wall = time.perf_counter()-t0
    if len(done) < len(pts):
        out(f"A2-CKPT stopped after {n_new} new points ({len(done)}/{len(pts)} in the checkpoint; --max-new {max_new}); re-run to resume")
        return None
    res = []
    for k in range(len(pts)):
        recs = [dict(r) for r in done[k]['recs']]
        for r in recs:
            if 'J' in r: r['J'] = np.array(r['J'], dtype=float)
        res.append((k, recs))
    lines = []; Js = {}
    for k, recs in res:
        sod = recs[0]; assert sod['table'] == 'SoD'
        for r in recs:
            r['id'] = f"{NAME}|{r['table']}|{point_id(r)}"
            r['selection26'] = bool(in_selection(r)); r['m'] = M_OF.get(r['table'])
            if r['status'] == 'ok':
                Js[r['id']] = r.pop('J')
                r['budget_ok'] = bool(r['EN'] <= B); r['budget_label'] = VERDICT; r['slack'] = B-r['EN']
                if r['table'] == 'SoD': r['dmD'] = 0.0
                elif sod['status'] == 'ok':
                    r['logdet_SoD'] = sod['logdet']; r['EN_SoD'] = sod['EN']; r['dmD'] = r['logdet']-sod['logdet']
                else: r['dmD'] = None
            else:
                r.pop('J', None)
            lines.append(r)
    nom = {r['table']: r for r in lines if r['cls'] == 'theta0'}
    for r in lines:
        if r['status'] == 'ok' and r['table'] != 'SoD' and r['dmD'] is not None:
            r['dmD_nominal'] = nom[r['table']]['dmD']; r['retention'] = r['dmD']/r['dmD_nominal']
    with open(OUT['archive'], 'w') as f:
        for r in lines: f.write(json.dumps(r, ensure_ascii=False)+'\n')
    np.savez(OUT['jnpz'], **Js)
    hdr = (f"{'table':5s} {'cls':6s} {'dir':20s} {'sg':2s} {'eps_t':>5s} {'eps_att':>12s} {'sel':>3s} {'E[N]':>10s} {'S':>10s} {'logdet':>11s} "
           f"{'dmD':>9s} {'keep':>7s} {'E[N]<=B':>8s} {'B-E[N]':>9s} {'bnd':>4s}")
    out(f"A2-TAB {hdr}")
    for r in lines:
        sg = '+' if r['sign'] > 0 else '-'
        if r['status'] != 'ok':
            out(f"A2-TAB {r['table']:5s} {r['cls']:6s} {r['dir']:20s} {sg:2s} {r['eps_target']:5.2f} STOP {r['message']}"); continue
        out(f"A2-TAB {r['table']:5s} {r['cls']:6s} {r['dir']:20s} {sg:2s} {r['eps_target']:5.2f} {r['eps_attained']:12.9f} "
            f"{('yes' if r['selection26'] else ''):>3s} {r['EN']:10.4f} {r['S']:10.3f} {r['logdet']:11.5f} "
            f"{(r['dmD'] if r['dmD'] is not None else float('nan')):+9.4f} {r.get('retention', float('nan')):7.3f} "
            f"{('yes' if r['budget_ok'] else 'NO'):>8s} {r['slack']:+9.4f} {r.get('boundary_steps', 0):4d}")
    ok = [r for r in lines if r['status'] == 'ok']
    out(f"A2-SUM evaluations {len(lines)} (ok {len(ok)}, assertion stops {len(lines)-len(ok)}); wall {wall:.1f} s with {workers} workers; "
        f"sum of per-evaluation wall {sum(r['wall_s'] for r in lines):.1f} s")
    out(f"A2-SUM max |sum m - 1| {max(r['mass_dev_max'] for r in ok)!r}; min m {min(r['min_m'] for r in ok)!r}; assertion failures "
        f"{sum(r['assert_fail'] for r in ok)}; invariants S, J, w all true: {all(r['inv_S'] and r['inv_J'] and r['inv_w'] for r in ok)}")
    out(f"A2-SUM table evaluations using a boundary lookup with mass: {sum(1 for r in ok if r['table'] != 'SoD' and r['boundary_steps'] > 0)}")
    reg = regression(out, lines, Js)
    json.dump(dict(wall_s=wall, workers=workers, n=len(lines), n_ok=len(ok), n_stop=len(lines)-len(ok), regression=reg),
              open(OUT['sweep'], 'w'), indent=1)
    return lines

def regression(out, lines, Js):
    """check 1 (theta_0, bit for bit vs round 5 and round 4) and the 80-point regression of m000 and the SoD vs round 5."""
    chk5 = json.load(open(os.path.join(R5, 'results', 'partA_check.json')))['results']
    nom = {r['table']: r for r in lines if r['cls'] == 'theta0'}
    res = {}
    for t, key, jref in (('m000', 'U/W', 'J_U_W.npy'), ('SoD', 'U/SoD', 'J_U_SoD.npy')):
        r = nom[t]; ref = chk5[key]
        eq = {k: r[k] == ref[k] for k in ('EN', 'S', 'logdet')}
        if t == 'm000': eq['dmD'] = r['dmD'] == ref['dmD']
        eq['J_round4'] = bool(np.array_equal(Js[r['id']], np.load(os.path.join(R4, 'results', jref))))
        res[f'theta0:{t}'] = eq
        for k, v in eq.items():
            out(f"A2-CHK1 {t} at theta_0 {k}: {r.get(k, '(81 J elements vs round4/results/'+jref+')') if k != 'J_round4' else '81 J elements vs round4/results/'+jref} "
                f"{'' if k == 'J_round4' else '(round 5 '+repr(ref[k])+') '}equal {v}")
    ok1 = all(all(v.values()) for v in res.values())
    out(f"A2-CHK1 spec §4 item 1 (m = 0 table and the SoD at theta_0 equal round 5 / round 4 bit for bit): {'PASS' if ok1 else 'FAIL'}")
    # all 80 points: m000 and SoD vs round 5's archive (U/W, U/SoD rows), E[N], E[N]_P, S, log det, J
    a5 = {r['id']: r for r in (json.loads(l) for l in open(os.path.join(R5, 'results', 'archive_partA.jsonl'))) if r['input'] == 'U'}
    J5 = np.load(os.path.join(R5, 'results', 'J_partA.npz'))
    n = 0; bad = []
    for r in lines:
        if r['cls'] == 'theta0' or r['table'] not in ('m000', 'SoD'): continue
        id5 = f"U|{'W' if r['table'] == 'm000' else 'SoD'}|{r['cls']}|{r['dir']}|{'+' if r['sign'] > 0 else '-'}|{r['eps_target']}"
        q = a5[id5]; n += 1
        same = all(r[k] == q[k] for k in ('EN', 'EN_P', 'S', 'logdet', 'eps_attained')) and r['theta'] == q['theta'] \
            and bool(np.array_equal(Js[r['id']], J5[id5]))
        if r['table'] == 'm000': same = same and r['dmD'] == q['dmD']
        if not same: bad.append(r['id'])
    res['round5_80'] = dict(n=n, mismatches=bad)
    out(f"A2-CHK1b m000 and SoD at the 80 points vs round5/results/archive_partA.jsonl and J_partA.npz (E[N], E[N]_P, S, log det, "
        f"Delta m_D, theta, eps_attained, 81 J; exact): {n - len(bad)}/{n} equal; mismatches {bad}")
    res['pass'] = ok1; res['pass_80'] = not bad
    return res


# --- summary (A3-A5) ----------------------------------------------------------------------------------------------------------
def read_archive(): return [json.loads(l) for l in open(ARCHIVE)]

def nondecreasing(v): return all(v[i] >= v[i-1] for i in range(1, len(v)))
def nonincreasing(v): return all(v[i] <= v[i-1] for i in range(1, len(v)))
def shape(v):
    up, dn = nondecreasing(v), nonincreasing(v)
    return '常数' if (up and dn) else ('单调上升' if up else ('单调下降' if dn else '非单调'))

def loc(r): return f"{r['cls']} {r['dir']} {'+' if r['sign'] > 0 else '-'}{r['eps_target']}"

def summary(out):
    A = [r for r in read_archive()]
    assert all(r['status'] == 'ok' for r in A), "assertion stop in the archive"
    fin = json.load(open(os.path.join(RES, 'margin_final.json')))
    rows = {}
    for t in TABS:
        q = [r for r in A if r['table'] == t]; nom = [r for r in q if r['cls'] == 'theta0'][0]
        p80 = [r for r in q if r['cls'] != 'theta0']; p26 = [r for r in p80 if r['selection26']]
        assert len(p80) == 80 and len(p26) == 26
        w26 = min(p26, key=lambda r: r['dmD']); w80 = min(p80, key=lambda r: r['dmD'])
        x26 = max(p26, key=lambda r: r['EN']); x80 = max(p80, key=lambda r: r['EN'])
        n26 = sum(r['budget_ok'] for r in p26)
        gm = [nom['dmD']]+[float(np.median([r['dmD'] for r in p80 if r['cls'] == 'P4' and r['eps_target'] == e])) for e in EPS_P4]
        d = dict(m=M_OF[t], B_m=fin[t]['B_m'], EN0=nom['EN'], slack_Bm0=fin[t]['B_m']-nom['EN'], dmD0=nom['dmD'], logdet0=nom['logdet'],
                 n26_within_B=n26, budget_pass26=bool(n26 == 26),
                 worst_gain26=w26['dmD'], worst_gain26_at=loc(w26), rho26=w26['retention'], info_label=('信息保持' if w26['retention'] >= 0.8 else '信息保持未达'),
                 worst_gain80=w80['dmD'], worst_gain80_at=loc(w80), rho80=w80['retention'],
                 maxEN26=x26['EN'], maxEN26_at=loc(x26), min_slack26=B-x26['EN'],
                 maxEN80=x80['EN'], maxEN80_at=loc(x80), min_slack80=B-x80['EN'], n80_within_B=sum(r['budget_ok'] for r in p80),
                 over26=[loc(r) for r in p26 if not r['budget_ok']],
                 gmed=gm, gmed_shape_4=shape(gm[1:]), gmed_shape_with0=shape(gm))
        for cls in ('P1', 'P2', 'P3', 'P4'):
            c = [r for r in p80 if r['cls'] == cls]
            d[f'n_{cls}_within_B'] = f"{sum(r['budget_ok'] for r in c)}/{len(c)}"
        # M2 (iv) points: P4 eps = 1 (ten directions) and P1 +-1
        m2 = [r for r in p80 if (r['cls'] == 'P4' and r['eps_target'] == 1.0) or (r['cls'] == 'P1' and r['eps_target'] == 1.0)]
        d['n_P4eps1_P1pm1_within_B'] = f"{sum(r['budget_ok'] for r in m2)}/{len(m2)}"
        d['P4eps1_P1pm1_over'] = [loc(r) for r in m2 if not r['budget_ok']]
        rows[t] = d
        for k, v in d.items(): out(f"A5 {t} {k} {v!r}")
    out(f"A5-TAB {'table':5s} {'m':>5s} {'E[N](th0)':>10s} {'dmD(th0)':>9s} {'26 ok':>5s} {'worst26':>8s} {'rho26':>6s} {'worst80':>8s} "
        f"{'maxEN26':>9s} {'slack26':>8s} {'maxEN80':>9s} {'slack80':>9s}  g_med(0, 0.2, 0.5, 1, 2)")
    for t, d in rows.items():
        out(f"A5-TAB {t:5s} {d['m']:5.2f} {d['EN0']:10.4f} {d['dmD0']:9.5f} {d['n26_within_B']:2d}/26 {d['worst_gain26']:8.4f} {d['rho26']:6.4f} "
            f"{d['worst_gain80']:8.4f} {d['maxEN26']:9.4f} {d['min_slack26']:+8.4f} {d['maxEN80']:9.3f} {d['min_slack80']:+9.3f}  "
            f"{', '.join(f'{v:.4f}' for v in d['gmed'])} ({d['gmed_shape_4']} over 0.2-2; {d['gmed_shape_with0']} with 0)")
    passing = [t for t in TABS[1:] if rows[t]['budget_pass26']]
    mstar = min(passing, key=lambda t: M_OF[t]) if passing else None
    sel = mstar if mstar else 'm020'
    out(f"A3 m = 0 table (not a candidate) 26-point budget pass: {rows['m000']['budget_pass26']} ({rows['m000']['n26_within_B']}/26)")
    out(f"A3 margins passing the 26-point budget condition ({VERDICT}): {passing}")
    if mstar:
        out(f"A3 m* = {M_OF[mstar]} ({mstar}); rho_26 = {rows[mstar]['rho26']!r} -> {rows[mstar]['info_label']}; nominal cost "
            f"Delta m_D(0; theta_0) - Delta m_D(m*; theta_0) = {rows['m000']['dmD0']-rows[mstar]['dmD0']!r} nat")
    else:
        out(f"A3 受测档位未找到满足条件的表; m* undefined; Parts B and C are carried out on m = 0.20 and labelled as such")
    for t in TABS:
        out(f"A4 {t} rho_26 {rows[t]['rho26']!r} ({rows[t]['info_label']}); worst at {rows[t]['worst_gain26_at']}")
    res = dict(rows=rows, passing=passing, mstar=mstar, selected=sel, selected_label=('m*' if mstar else '受测档位未找到满足条件的表; m = 0.20'),
               nominal_cost=(rows['m000']['dmD0']-rows[mstar]['dmD0']) if mstar else None)
    json.dump(res, open(SUMMARY, 'w'), indent=1, ensure_ascii=False)
    return res


# --- figures ------------------------------------------------------------------------------------------------------------------
def figs(out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    A = read_archive(); S_ = json.load(open(SUMMARY)); rows = S_['rows']
    INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    COL = {'m000': '#86b6ef', 'm005': '#3987e5', 'm010': '#1c5cab', 'm020': '#0d366b'}   # ordinal blue ramp (validated, light surface)
    MK = {'m000': 'o', 'm005': 's', 'm010': '^', 'm020': 'D'}                             # secondary encoding of m
    LAB = {t: f"m = {M_OF[t]:.2f}" + (' (round 4)' if t == 'm000' else '') for t in TABS}
    plt.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                         'axes.facecolor': SURF, 'figure.facecolor': SURF, 'font.size': 8.5, 'axes.titlesize': 9})
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.6), sharey='row', constrained_layout=True)
    for i, (qty, ylab) in enumerate((('EN', r'$\mathbb{E}[N]$ (discrete evaluator)'), ('dmD', r'$\Delta m_D$ vs SoD at the same $\theta$ (nat)'))):
        for t in TABS:
            q = [r for r in A if r['table'] == t]; nom = [r for r in q if r['cls'] == 'theta0'][0][qty]
            # P1 signed eps
            rr = sorted([r for r in q if r['cls'] == 'P1'], key=lambda r: r['sign']*r['eps_attained'])
            xs = [0.0]+[r['sign']*r['eps_attained'] for r in rr]; ys = [nom]+[r[qty] for r in rr]; o = np.argsort(xs)
            axs[i, 0].plot(np.array(xs)[o], np.array(ys)[o], '-', marker=MK[t], color=COL[t], lw=1.8, ms=4.5, label=LAB[t])
            # P4 median and range
            med = [nom]+[float(np.median([r[qty] for r in q if r['cls'] == 'P4' and r['eps_target'] == e])) for e in EPS_P4]
            lo = [nom]+[min(r[qty] for r in q if r['cls'] == 'P4' and r['eps_target'] == e) for e in EPS_P4]
            hi = [nom]+[max(r[qty] for r in q if r['cls'] == 'P4' and r['eps_target'] == e) for e in EPS_P4]
            ex = [0.0]+list(EPS_P4)
            axs[i, 1].plot(ex, med, '-', marker=MK[t], color=COL[t], lw=1.8, ms=4.5, label=LAB[t])
            axs[i, 1].fill_between(ex, lo, hi, color=COL[t], alpha=0.10, lw=0)
            # P2 signed eps
            rr = sorted([r for r in q if r['cls'] == 'P2'], key=lambda r: r['sign']*r['eps_attained'])
            xs = [0.0]+[r['sign']*r['eps_attained'] for r in rr]; ys = [nom]+[r[qty] for r in rr]; o = np.argsort(xs)
            axs[i, 2].plot(np.array(xs)[o], np.array(ys)[o], '-', marker=MK[t], color=COL[t], lw=1.8, ms=4.5, label=LAB[t])
        for j, ax in enumerate(axs[i]):
            if qty == 'EN': ax.axhline(B, color='#e34948', lw=1.0, ls='--', zorder=0)
            else: ax.axhline(0.0, color=INK2, lw=0.6, zorder=0)
            if j != 1:
                ax.set_xscale('symlog', linthresh=0.1); tk = [-4, -2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2, 4] if j == 0 else [-2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2]
                ax.set_xticks(tk); ax.set_xticklabels([f'{v:g}' for v in tk]); ax.minorticks_off(); ax.set_xlabel(r'signed $\varepsilon$ (attained)')
            else:
                ax.set_xticks([0, 0.2, 0.5, 1, 2]); ax.set_xlabel(r'$\varepsilon$')
            ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
            for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        axs[i, 0].set_ylabel(ylab)
        if qty == 'EN':
            for ax in axs[i]: ax.set_yscale('log'); ax.set_yticks([150, 200, 250, 300, 400, 600, 1000, 1500]); ax.set_yticklabels(['150', '200', '250', '300', '400', '600', '1000', '1500'])
    axs[0, 0].set_title(r'P1  $\theta_1+\varepsilon\sigma$', loc='left', color=INK)
    axs[0, 1].set_title('P4  median over ten directions (band: min-max)', loc='left', color=INK)
    axs[0, 2].set_title(r'P2  $\theta_{2..9}\times(1+\eta)$', loc='left', color=INK)
    axs[0, 0].legend(frameon=False, fontsize=7.5, loc='upper left', labelcolor=INK)
    fig.suptitle(f"Budget-margin tables under model mismatch (uniform input, weighted objective; discrete evaluation, sigma/24 grid). "
                 f"Dashed red: true budget B = {B:.4f}", color=INK, fontsize=10, x=0.01, ha='left')
    fig.savefig(os.path.join(HERE, 'fig_margin_EN_dmD.png'), dpi=130); plt.close(fig)
    out("A-FIG fig_margin_EN_dmD.png written")
    # nominal gain and minimum slack vs m
    ms = [M_OF[t] for t in TABS]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axs[0].plot(ms, [rows[t]['dmD0'] for t in TABS], '-o', color='#2a78d6', lw=2, ms=5, label=r'$\Delta m_D(\theta_0)$')
    axs[0].plot(ms, [rows[t]['worst_gain26'] for t in TABS], '-s', color='#6250d6', lw=1.4, ms=4.5, label='worst over the 26 points')
    axs[0].set_ylabel('nat'); axs[0].set_title(r'Nominal gain $\Delta m_D(\theta_0)$ vs the $\Delta^*$ SoD', loc='left', color=INK)
    axs[0].legend(frameon=False, fontsize=7.5, labelcolor=INK)
    axs[1].plot(ms, [rows[t]['min_slack26'] for t in TABS], '-o', color='#eb6834', lw=2, ms=5, label=r'min over the 26 points of $B-\mathbb{E}[N]$')
    axs[1].plot(ms, [rows[t]['slack_Bm0']+B-rows[t]['B_m'] for t in TABS], '-s', color='#1baf7a', lw=1.4, ms=4.5, label=r'$B-\mathbb{E}[N](\theta_0)$')
    axs[1].axhline(0.0, color='#e34948', lw=1.0, ls='--', zorder=0)
    axs[1].set_ylabel('sends'); axs[1].set_title('Budget slack against the true B (discrete evaluator)', loc='left', color=INK)
    axs[1].legend(frameon=False, fontsize=7.5, labelcolor=INK)
    for ax in axs:
        ax.set_xticks(ms); ax.set_xlabel('margin m (design target B_m = (1 - m) B)')
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    fig.savefig(os.path.join(HERE, 'fig_margin_cost_slack.png'), dpi=130); plt.close(fig)
    out("A-FIG fig_margin_cost_slack.png written")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['sweep', 'summary', 'figs']); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--outdir'); ap.add_argument('--max-new', type=int)
    A_ = ap.parse_args()
    if A_.outdir: set_outdir(A_.outdir)
    sys.stdout = dr.Tee(os.path.join(A_.outdir or LOGS, f'partA_{A_.stage}.log'), 'a' if A_.outdir else 'w')
    import numba, scipy
    print(f"=== robustness_margin.py {A_.stage} started {time.strftime('%Y-%m-%d %H:%M:%S')}; spec reports_CN/plan_round6_CN.md v3.2; python "
          f"{sys.version.split()[0]}, numpy {np.__version__}, scipy {scipy.__version__}, numba {numba.__version__}; B = {B!r}")
    t0 = time.perf_counter()
    {'sweep': lambda: sweep(print, A_.workers, A_.max_new), 'summary': lambda: summary(print), 'figs': lambda: figs(print)}[A_.stage]()
    print(f"=== robustness_margin.py {A_.stage} done in {time.perf_counter()-t0:.1f} s")
