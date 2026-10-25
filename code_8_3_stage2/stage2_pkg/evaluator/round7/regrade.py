"""Round 7, Part C: fine-grid regrade of the five candidates (spec reports_CN/plan_round7_CN.md v3.1, frozen; §1.3, §2.2
items 1-3; task C1-C5). 固定设计、离散方案下 -- discrete evaluations of fixed tables on three quadrature grids, not bounds.

Candidates: m000 (U_W_r05), m005, m010, m020 (round 6 margin tables), T4 (U_W_m020_r02, lambda/gbar = 4); control: the SoD at
Delta* (M = 480). Points: theta_0 + the round-6 26-point selection set (exact theta from the round-6 archive). Grids: f = 1
(sigma/24), 3 (sigma/72), 5 (sigma/120); M always in sigma/24 units, f M grid steps on the grid of factor f.

  run      C1: T4 at the 27 points at f = 1 (its nominal value regressed against the scan record); the four old tables and the
           SoD at f = 1 are the round-6 archive values, recomputed here only as a regression (bit for bit);
           C2: five candidates + SoD at the 27 points at f = 3 and f = 5 (324 evaluations).
           Every evaluation is appended (fsync) to results/archive_regrade.jsonl as it completes (J as exact JSON floats);
           a re-run resumes from it. 4 workers.
  summary  C3 convergence (signed d_1 = N_24 - N_72, d_2 = N_72 - N_120, ratio, sign, gate max|d_2| <= 0.05 over the five
           candidates x 27 points), C4 budget / rho_26 / c on each grid and the f = 3 -> 5 changes, C5 精度未定 flags, the
           selection rule of spec §2.4 item 3 -> results/partC_regrade.json, logs/partC_summary.log
  figs     fig_grid_EN_logdet.png (E[N] and log det vs grid per candidate), fig_d1_d2.png (signed d_1 vs d_2)

Usage (from this directory): python3 regrade.py run | summary | figs
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import evaluator_refine as er
ew, ef, dr, rb = er.ew, er.ef, er.dr, er.rb
B = er.B
ARCH = os.path.join(er.RES, 'archive_regrade.jsonl'); OUTJ = os.path.join(er.RES, 'partC_regrade.json')
CANDS = er.CANDIDATES; ALL = CANDS+('SoD',)
GATE = 0.05; NEAR = 1e-8; PREC = 0.05
VERDICT = '离散方案下'

_PTS = None; _TABS = None
def _init():
    global _PTS, _TABS
    er.warmup(); _PTS = {p['pid']: p for p in er.points27()}; _TABS = er.load_tables()

def _job(job):
    cand, pid, f = job
    t0 = time.perf_counter()
    try:
        tab = er.SodTable(480) if cand == 'SoD' else _TABS[cand]
        rec, J, r = er.evaluate(tab, f, _PTS[pid]['theta'])
        d = er.record(rec, J); d['status'] = 'ok'
    except (ew.AssertionStop, ef.AssertionStop, er.BoundaryNode) as e:
        d = dict(status='assert_stop', message=str(e))
    d.update(cand=cand, pid=pid, f=f, wall_s=time.perf_counter()-t0)
    return d

def read_archive():
    A = {}
    if os.path.exists(ARCH):
        for l in open(ARCH):
            d = json.loads(l); A[(d['cand'], d['pid'], d['f'])] = d
    return A

def run(out, workers=4):
    import multiprocessing as mp
    pts = er.points27(); A = read_archive()
    jobs = [(c, p['pid'], f) for f in (5, 3) for c in ALL for p in pts]
    jobs += [(c, p['pid'], 1) for c in CANDS for p in pts]+[('SoD', p['pid'], 1) for p in pts]
    todo = [j for j in jobs if j not in A]
    out(f"C0 {len(pts)} points x (five candidates + SoD) x f in (1, 3, 5): {len(jobs)} evaluations; archive {os.path.relpath(ARCH, HERE)}: "
        f"{len(jobs)-len(todo)} resumed, {len(todo)} to evaluate; {workers} workers")
    t0 = time.perf_counter(); n = 0
    if todo:
        with mp.get_context('fork').Pool(workers, initializer=_init) as pool, open(ARCH, 'a') as fa:
            for d in pool.imap_unordered(_job, todo, chunksize=1):
                fa.write(json.dumps(d, ensure_ascii=False)+'\n'); fa.flush(); os.fsync(fa.fileno()); n += 1
                if d['status'] != 'ok': out(f"C0 ASSERTION STOP {d['cand']} {d['pid']} f={d['f']}: {d['message']}")
                if n % 25 == 0 or n == len(todo):
                    out(f"C0 progress {n}/{len(todo)} ({time.perf_counter()-t0:.0f} s); last {d['cand']} {d['pid']} f={d['f']} E[N] {d.get('EN')!r} {d['wall_s']:.1f} s")
    wall = time.perf_counter()-t0
    out(f"C0 run wall {wall:.1f} s ({n} new evaluations)")
    return wall


# --- summary ------------------------------------------------------------------------------------------------------------
def table_of_values():
    """N[c][pid][f], L[c][pid][f] (log det), with f = 1 for the old tables and the SoD from the round-6 archive (spec C1);
    regression of the f = 1 recomputations against it."""
    A = read_archive(); pts = er.points27(); pids = [p['pid'] for p in pts]
    assert all(d['status'] == 'ok' for d in A.values()), "assertion stop in the archive"
    A6 = {}
    for l in open(os.path.join(er.R6, 'results', 'archive_partA.jsonl')):
        r = json.loads(l)
        pid = 'theta_0' if r['cls'] == 'theta0' else f"{r['cls']}|{r['dir']}|{'+' if r['sign'] > 0 else '-'}|{r['eps_target']}"
        A6[(r['table'], pid)] = r
    J6 = np.load(os.path.join(er.R6, 'results', 'J_partA.npz'))
    N = {c: {p: {} for p in pids} for c in ALL}; L = {c: {p: {} for p in pids} for c in ALL}; W = {c: {p: {} for p in pids} for c in ALL}
    reg_bad = []; nreg = 0
    for c in ALL:
        for p in pids:
            for f in (3, 5):
                d = A[(c, p, f)]; N[c][p][f] = d['EN']; L[c][p][f] = d['logdet']; W[c][p][f] = d['wall_s']
            d1 = A[(c, p, 1)]; W[c][p][1] = d1['wall_s']
            if c == 'T4':
                N[c][p][1] = d1['EN']; L[c][p][1] = d1['logdet']
            else:
                r6 = A6[(c, p)]; N[c][p][1] = r6['EN']; L[c][p][1] = r6['logdet']; nreg += 1
                same = d1['EN'] == r6['EN'] and d1['S'] == r6['S'] and d1['logdet'] == r6['logdet'] and d1['EN_P'] == r6['EN_P'] \
                    and np.array_equal(np.array(d1['J']), J6[r6['id']])
                if not same: reg_bad.append((c, p))
    return A, pts, pids, N, L, W, dict(n=nreg, mismatches=reg_bad)

def summary(out):
    A, pts, pids, N, L, W, reg = table_of_values()
    P26 = [p['pid'] for p in pts if p['sel26']]; P0 = 'theta_0'
    lab = er.label
    out(f"C1-REG the four old tables and the SoD at f = 1, recomputed by the refined evaluator at the 27 points, vs round 6's archive "
        f"(E[N], E[N]_P, S, log det, 81 J; exact): {reg['n']-len(reg['mismatches'])}/{reg['n']} equal; mismatches {reg['mismatches']}")
    sc = json.load(open(os.path.join(er.R6, 'results', 'scan_U_W_m020.json'))); r2 = [r for r in sc['runs'] if r['run'] == 2][0]
    t4 = A[('T4', P0, 1)]; ld_sod = L['SoD'][P0][1]
    t4reg = dict(EN=t4['EN'] == r2['EN'], S=t4['S'] == r2['S'], logdet=t4['logdet'] == r2['logdet'], dmD=t4['logdet']-ld_sod == r2['dmD'])
    out(f"C1-REG T4 at theta_0, f = 1, vs scan_U_W_m020.json run 2 (exact): {t4reg}; E[N] {t4['EN']!r}, Delta m_D {t4['logdet']-ld_sod!r}")
    # one line per number: E[N], log det, Delta m_D on the three grids
    dmD = {c: {p: {f: L[c][p][f]-L['SoD'][p][f] for f in (1, 3, 5)} for p in pids} for c in ALL}
    for c in ALL:
        for p in pids:
            for f in (1, 3, 5):
                out(f"C2 {c} {lab(p)} f={f} E[N] {N[c][p][f]!r} logdet {L[c][p][f]!r} dmD {dmD[c][p][f]!r}")
    # C3 convergence
    conv = {}; rows = []
    for c in ALL:
        for p in pids:
            d1 = N[c][p][1]-N[c][p][3]; d2 = N[c][p][3]-N[c][p][5]
            ratio = d1/d2 if abs(d2) > NEAR else None
            und = (np.sign(d1) != np.sign(d2)) or abs(d2) <= NEAR
            conv[(c, p)] = dict(d1=d1, d2=d2, ratio=ratio, sign_consistent=bool(np.sign(d1) == np.sign(d2) and abs(d2) > NEAR),
                                order_undetermined=bool(und))
            out(f"C3 {c} {lab(p)} d1 {d1!r} d2 {d2!r} ratio {ratio!r} sign_consistent {conv[(c, p)]['sign_consistent']} "
                f"{'阶数未定' if und else ''}")
    cpairs = [(c, p) for c in CANDS for p in pids]
    d1max = max(cpairs, key=lambda k: abs(conv[k]['d1'])); d2max = max(cpairs, key=lambda k: abs(conv[k]['d2']))
    ratios = [conv[k]['ratio'] for k in cpairs if conv[k]['ratio'] is not None]
    med_ratio = float(np.median(ratios)) if ratios else None
    n_cons = sum(conv[k]['sign_consistent'] for k in cpairs); n_und = sum(conv[k]['order_undetermined'] for k in cpairs)
    gate = abs(conv[d2max]['d2']) <= GATE
    out(f"C3 max|d1| over the five candidates x 27 points {abs(conv[d1max]['d1'])!r} at {d1max[0]} {lab(d1max[1])} (signed {conv[d1max]['d1']!r})")
    out(f"C3 max|d2| over the five candidates x 27 points {abs(conv[d2max]['d2'])!r} at {d2max[0]} {lab(d2max[1])} (signed {conv[d2max]['d2']!r})")
    out(f"C3 ratio d1/d2 (diagnostic; first order ~5, second order ~12.5): median {med_ratio!r} over {len(ratios)} pairs, min {min(ratios)!r}, "
        f"max {max(ratios)!r}; sign consistent {n_cons}/{len(cpairs)}; 阶数未定 {n_und}/{len(cpairs)}")
    for c in CANDS:
        q = [conv[(c, p)] for p in pids]
        out(f"C3 {c}: max|d1| {max(abs(x['d1']) for x in q)!r}, max|d2| {max(abs(x['d2']) for x in q)!r}, median ratio "
            f"{float(np.median([x['ratio'] for x in q if x['ratio'] is not None]))!r}, sign consistent {sum(x['sign_consistent'] for x in q)}/27, "
            f"d1 range [{min(x['d1'] for x in q)!r}, {max(x['d1'] for x in q)!r}], d2 range [{min(x['d2'] for x in q)!r}, {max(x['d2'] for x in q)!r}]")
    q = [conv[('SoD', p)] for p in pids]
    out(f"C3 SoD at Delta* (control, not in the gate): max|d1| {max(abs(x['d1']) for x in q)!r}, max|d2| {max(abs(x['d2']) for x in q)!r}")
    out(f"C3 GATE max|d2| <= {GATE} send over the five candidates x 27 points: {'PASS -> sigma/120 is the reference grid' if gate else 'FAIL -> 网格未收敛 (verdict case 1)'}; "
        f"two grids agreeing does not prove closeness to the continuous model")
    # C4 per grid
    grid_rows = {}
    for f in (1, 3, 5):
        for c in CANDS:
            n26 = [N[c][p][f] for p in P26]; over = [lab(p) for p in P26 if N[c][p][f] > B]
            ret = {p: dmD[c][p][f]/dmD[c][P0][f] for p in P26}
            wp = min(P26, key=lambda p: ret[p]); xp = max(P26, key=lambda p: N[c][p][f]); gp = min(P26, key=lambda p: dmD[c][p][f])
            grid_rows[(c, f)] = dict(EN0=N[c][P0][f], logdet0=L[c][P0][f], dmD0=dmD[c][P0][f], c=L['m000'][P0][f]-L[c][P0][f],
                                     n26_within=sum(v <= B for v in n26), budget_pass26=bool(all(v <= B for v in n26)), over26=over,
                                     rho26=ret[wp], rho26_at=lab(wp), maxEN26=N[c][xp][f], maxEN26_at=lab(xp), min_slack26=B-N[c][xp][f],
                                     min_dmD26=dmD[c][gp][f], min_dmD26_at=lab(gp))
            for k, v in grid_rows[(c, f)].items(): out(f"C4 {c} f={f} {k} {v!r}")
    for c in CANDS:
        a, b = grid_rows[(c, 3)], grid_rows[(c, 5)]
        dl27 = max(abs(L[c][p][5]-L[c][p][3]) for p in pids)
        out(f"C4-CHG {c} f=3 -> f=5: log det(theta_0) {a['logdet0']!r} -> {b['logdet0']!r} (change {b['logdet0']-a['logdet0']!r}); "
            f"c {a['c']!r} -> {b['c']!r} (change {b['c']-a['c']!r}); rho_26 {a['rho26']!r} -> {b['rho26']!r} (change {b['rho26']-a['rho26']!r}); "
            f"max over 27 points |log det_120 - log det_72| {dl27!r}")
    dl_all = max(abs(L[c][p][5]-L[c][p][3]) for c in ALL for p in pids)
    out(f"C4-CHG max over the five candidates and the SoD, 27 points, |log det_120 - log det_72| {dl_all!r} nat")
    # C5 precision flags (sigma/72 vs sigma/120)
    prec = {}
    for c in CANDS:
        for p in pids:
            n3, n5 = N[c][p][3], N[c][p][5]; rs = []
            if (n3 <= B) != (n5 <= B): rs.append('预算判定在 sigma/72 与 sigma/120 间翻转')
            if abs(conv[(c, p)]['d2']) > GATE: rs.append(f"|d2| = {abs(conv[(c, p)]['d2']):.4f} > {GATE}")
            if min(abs(n3-B), abs(n5-B)) <= PREC: rs.append(f'贴预算线 min(|N_72 - B|, |N_120 - B|) = {min(abs(n3-B), abs(n5-B)):.4f} <= {PREC}')
            prec[(c, p)] = rs
            if rs: out(f"C5 {c} {lab(p)} 精度未定: {rs}")
    for c in CANDS:
        out(f"C5 {c}: 精度未定 at {sum(bool(prec[(c, p)]) for p in pids)}/27 points; actually over B on sigma/120 at "
            f"{sum(N[c][p][5] > B for p in P26)}/26 selection points ({VERDICT})")
    # selection (spec §2.4 items 2-3), on the reference grid only
    sel = dict(gate=gate, reference_grid='sigma/120' if gate else None)
    if gate:
        passers = [c for c in CANDS if grid_rows[(c, 5)]['budget_pass26']]
        sel['passers'] = passers
        cheap = [c for c in passers if grid_rows[(c, 5)]['c'] <= 2.0]
        if passers: sel['min_c_passer'] = min(passers, key=lambda c: grid_rows[(c, 5)]['c']); sel['min_c'] = grid_rows[(sel['min_c_passer'], 5)]['c']
        if cheap:
            s_ = sorted(cheap, key=lambda c: (grid_rows[(c, 5)]['c'], -grid_rows[(c, 5)]['EN0']))[0]
            sel['selected'] = s_; sel['policy_id'] = json.load(open(os.path.join(er.RES, 'tables_round7.json')))['tables'][s_]['policy_id']
            g = grid_rows[(s_, 5)]
            xp = max(P26, key=lambda p: N[s_][p][5]); gp = min(P26, key=lambda p: dmD[s_][p][5])
            mains = {}
            for role, p in (('theta_0', P0), ('max E[N]_120 in the 26', xp), ('min Delta m_D,120 in the 26', gp)):
                mains.setdefault(p, []).append(role)
            sel['main_points'] = [dict(pid=p, label=lab(p), roles=r, EN72=N[s_][p][3], EN120=N[s_][p][5], d2=conv[(s_, p)]['d2'], flags=prec[(s_, p)],
                                       undetermined=bool(prec[(s_, p)])) for p, r in mains.items()]
            sel['case2'] = False
        else:
            sel['selected'] = None; sel['case2'] = True
        out(f"C4-SEL reference grid sigma/120: candidates passing the 26-point budget {passers}; with c <= 2: {cheap}; "
            f"selected {sel.get('selected')} ({sel.get('policy_id')}); case 2 (受测候选未找到低代价方案) {sel['case2']}")
        for mp_ in sel.get('main_points', []):
            out(f"C4-SEL main check point {mp_['label']} ({'; '.join(mp_['roles'])}): E[N]_72 {mp_['EN72']!r}, E[N]_120 {mp_['EN120']!r}, d2 {mp_['d2']!r}, "
                f"精度未定 {mp_['undetermined']} {mp_['flags']}")
    else:
        out("C4-SEL no reference grid (gate failed): no selection; the sigma/120 values above are description only (非参考网格)")
    walls = [W[c][p][5] for c in ALL for p in pids]
    out(f"C-TIME single-forward wall time sigma/120: median {float(np.median(walls)):.1f} s, max {max(walls):.1f} s; sigma/72 median "
        f"{float(np.median([W[c][p][3] for c in ALL for p in pids])):.1f} s; sigma/24 median {float(np.median([W[c][p][1] for c in ALL for p in pids])):.1f} s")
    res = dict(points=[dict(p, label=lab(p['pid'])) for p in pts], N={c: N[c] for c in ALL}, L={c: L[c] for c in ALL},
               dmD={c: dmD[c] for c in ALL}, conv={f'{c}|{p}': v for (c, p), v in conv.items()},
               gate=dict(max_d2=abs(conv[d2max]['d2']), at=list(d2max), max_d1=abs(conv[d1max]['d1']), d1_at=list(d1max), pass_=gate,
                         median_ratio=med_ratio, n_ratio=len(ratios), sign_consistent=n_cons, order_undetermined=n_und, n_pairs=len(cpairs)),
               grid_rows={f'{c}|{f}': v for (c, f), v in grid_rows.items()}, logdet_change_max_all=dl_all,
               precision={f'{c}|{p}': v for (c, p), v in prec.items()}, selection=sel, regression_f1=reg, T4_regression=t4reg,
               wall_median={f: float(np.median([W[c][p][f] for c in ALL for p in pids])) for f in (1, 3, 5)},
               wall_sum={f: float(sum(W[c][p][f] for c in ALL for p in pids)) for f in (1, 3, 5)})
    json.dump(res, open(OUTJ, 'w'), indent=1, ensure_ascii=False)
    # compact text tables
    out(f"C3-TAB {'cand':5s} {'point':22s} {'N_24':>10s} {'N_72':>10s} {'N_120':>10s} {'d1':>9s} {'d2':>9s} {'d1/d2':>8s}")
    for c in CANDS:
        for p in pids:
            v = conv[(c, p)]
            out(f"C3-TAB {c:5s} {lab(p):22s} {N[c][p][1]:10.4f} {N[c][p][3]:10.4f} {N[c][p][5]:10.4f} {v['d1']:+9.4f} {v['d2']:+9.5f} "
                f"{(v['ratio'] if v['ratio'] is not None else float('nan')):8.2f}{'  阶数未定' if v['order_undetermined'] else ''}")
    out(f"C4-TAB {'cand':5s} {'f':>2s} {'E[N](th0)':>10s} {'logdet(th0)':>12s} {'c':>8s} {'26 ok':>6s} {'max E[N]26':>10s} {'slack26':>8s} {'rho26':>7s} {'dmD(th0)':>9s}")
    for c in CANDS:
        for f in (1, 3, 5):
            g = grid_rows[(c, f)]
            out(f"C4-TAB {c:5s} {f:2d} {g['EN0']:10.4f} {g['logdet0']:12.6f} {g['c']:8.5f} {g['n26_within']:3d}/26 {g['maxEN26']:10.4f} "
                f"{g['min_slack26']:+8.4f} {g['rho26']:7.4f} {g['dmD0']:9.5f}")
    return res


# --- figures ------------------------------------------------------------------------------------------------------------
def figs(out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.load(open(OUTJ)); pids = [p['pid'] for p in R['points']]
    INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    # categorical slots 1-5 of the validated reference palette in fixed order (identity), SoD = neutral ink; the scatter has five
    # series on all pairs, beyond the three-slot all-pairs cap, so every series also carries its own marker shape (secondary encoding)
    COL = {'m000': '#2a78d6', 'm005': '#eb6834', 'm010': '#1baf7a', 'm020': '#eda100', 'T4': '#e87ba4', 'SoD': '#52514e'}
    MK = {'m000': 'o', 'm005': 's', 'm010': '^', 'm020': 'D', 'T4': 'v', 'SoD': 'x'}
    plt.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                         'axes.facecolor': SURF, 'figure.facecolor': SURF, 'font.size': 8.5, 'axes.titlesize': 9})
    hx = {1: 1/24, 3: 1/72, 5: 1/120}
    fig, axs = plt.subplots(2, 6, figsize=(16, 6.4), sharex=True, sharey='row', constrained_layout=True)
    for j, c in enumerate(ALL):
        for i, (q, ylab) in enumerate((('N', r'$\mathbb{E}[N]_h-\mathbb{E}[N]_{\sigma/120}$ (sends)'), ('L', r'$\log\det J_h-\log\det J_{\sigma/120}$ (nat)'))):
            ax = axs[i, j]
            for p in pids:
                v = R[q][c][p]; ys = [v[str(f)]-v['5'] for f in (1, 3, 5)]
                ax.plot([hx[f] for f in (1, 3, 5)], ys, '-o', color=COL[c], lw=(2.2 if p == 'theta_0' else 0.8), ms=(4.5 if p == 'theta_0' else 2.5),
                        alpha=(1.0 if p == 'theta_0' else 0.45), zorder=(3 if p == 'theta_0' else 2))
            ax.axhline(0.0, color=INK2, lw=0.6, zorder=0)
            if q == 'N': ax.axhspan(-GATE, GATE, color='#1baf7a', alpha=0.12, lw=0, zorder=0)
            ax.set_xticks([hx[f] for f in (1, 3, 5)]); ax.set_xticklabels(['σ/24', 'σ/72', 'σ/120'])
            ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
            for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
            if i == 0: ax.set_title(f"{c}" + (' (control)' if c == 'SoD' else ''), loc='left', color=INK)
            if j == 0: ax.set_ylabel(ylab)
            if i == 1: ax.set_xlabel('grid step h')
    fig.suptitle("Quadrature convergence of the frozen policies: change relative to sigma/120 at the 27 points (bold: theta_0; green band: |d| <= 0.05 send). "
                 "Fixed design, discrete evaluation.", color=INK, fontsize=10, x=0.01, ha='left')
    fig.savefig(os.path.join(HERE, 'fig_grid_EN_logdet.png'), dpi=130); plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.6), constrained_layout=True)
    for c in CANDS:
        d1 = [R['conv'][f'{c}|{p}']['d1'] for p in pids]; d2 = [R['conv'][f'{c}|{p}']['d2'] for p in pids]
        ax.scatter(d2, d1, s=34, marker=MK[c], color=COL[c], edgecolor=SURF, linewidth=0.8, label=c, zorder=3)
    D1 = [R['conv'][f'{c}|{p}']['d1'] for c in CANDS for p in pids]; D2 = [R['conv'][f'{c}|{p}']['d2'] for c in CANDS for p in pids]
    x0, x1 = min(0.0, min(D2)*1.15), max(0.0, max(D2)*1.15); y0, y1 = min(0.0, min(D1)*1.15), max(0.0, max(D1)*1.15)
    xx = np.array([x0, x1])
    for k, ls, lb in ((5.0, '--', 'd1 = 5 d2 (first order)'), (12.5, ':', 'd1 = 12.5 d2 (second order)')):
        ax.plot(xx, k*xx, ls, color=INK2, lw=0.9, label=lb, zorder=1)
    if x1 > GATE or x0 < -GATE:
        ax.axvspan(-GATE, GATE, color='#1baf7a', alpha=0.10, lw=0, label=f'gate |d2| <= {GATE}')
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.axhline(0, color=INK2, lw=0.5); ax.axvline(0, color=INK2, lw=0.5)
    ax.text(0.98, 0.03, f"gate |d2| <= {GATE} send; max |d2| = {max(abs(v) for v in D2):.4f} ({'inside' if max(abs(v) for v in D2) <= GATE else 'OUTSIDE'})",
            transform=ax.transAxes, ha='right', va='bottom', color=INK, fontsize=8)
    ax.set_xlabel(r'$d_2=N_{72}-N_{120}$ (sends)'); ax.set_ylabel(r'$d_1=N_{24}-N_{72}$ (sends)')
    ax.set_title('Signed grid differences, five candidates x 27 points', loc='left', color=INK)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc='best')
    fig.savefig(os.path.join(HERE, 'fig_d1_d2.png'), dpi=130); plt.close(fig)
    out("C-FIG fig_grid_EN_logdet.png, fig_d1_d2.png written")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['run', 'summary', 'figs']); ap.add_argument('--workers', type=int, default=4)
    A_ = ap.parse_args()
    sys.stdout = dr.Tee(os.path.join(er.LOGS, f'partC_{A_.stage}.log'), 'a' if A_.stage == 'run' else 'w')
    print(f"=== regrade.py {A_.stage} started {time.strftime('%Y-%m-%d %H:%M:%S')}; spec reports_CN/plan_round7_CN.md v3.1 §2.2; B = {B!r}")
    t0 = time.perf_counter()
    {'run': lambda: run(print, A_.workers), 'summary': lambda: summary(print), 'figs': lambda: figs(print)}[A_.stage]()
    print(f"=== regrade.py {A_.stage} done in {time.perf_counter()-t0:.1f} s")
