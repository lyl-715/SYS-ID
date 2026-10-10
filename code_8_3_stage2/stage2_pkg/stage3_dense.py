"""§8.5 stage 3, experiment 3: denser budgeted search.
(1) Local phase grids around NM-none-P5514 and AB-0.5: offsets {-60,-30,0,30,60} deg on all four phases (5^4 = 625 points each),
    powers fixed, R=200 (seed 20261010, the screening draws; reference (C) uniform on them), decision by qual (mc85_post.py).
(2) Nelder-Mead as in stage 2 (stage2_local.py's nm_run: budget penalty on, MU_B = 1 nat/send on max(0, D + 2se), MU_E = 20,
    same parametrisation and simplex), maxfev=200, R=500 inside (seed 20261010 extended), from
    - the best qualified point of each grid: for the NM-none-P5514 grid the qualified point with the largest Delta m_D (target 'none');
      for the AB-0.5 grid the qualified point with m_E >= 0.5 and the largest Delta m_D (target 0.5);
    - 4 additional starts per lambda0 in {none, 0.3, 0.5, 0.7} from the stage-2 R=2000 front, excluding every stage-2 start
      (stage2/local_runs.json) and both grid centres: per target, the eligible front point with the smallest R=2000 penalised objective
      in each front region (A: m_E<0.1, B: 0.1<=m_E<0.3, C: m_E>=0.3), plus the next-best eligible point overall.
(3) Re-evaluation at R=2000 (seed 20261012; same draws and reference as the stage-2 pool) of every run endpoint, of the per-target
    best penalised point over the archives of all stage-3 runs (as in stage 2), and of the 20 best points of each grid by that grid's
    R=200 penalised objective. Provenance per new candidate: origin_run and selected_by, as in stage 2.
(4) Empirical Pareto front of the evaluated candidates over the union of the stage-2 pool (54) and the new points; per floor, the
    stage-2 best vs the union best with paired batch se.
Run from code_8_3_stage2/stage2_pkg/:  python3 stage3_dense.py
Outputs (stage3/): stage3_dense.log, dense_grids.json, dense_runs.json, dense_R2000.json, front_union_R2000.json, fig_front_union.png"""
exec(open('stage3_common.py').read())
import itertools
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
T_START = time.time(); sys.stdout = Tee(f'{OUT}/stage3_dense.log')
print(f"=== stage3_dense.py: phase grids at R={R200} (seed {SEED200}); Nelder-Mead maxfev={MAXFEV}, R={RS} inside (seed {SEEDS}), MU_B={MU_B}, MU_E={MU_E}; "
      f"re-evaluation R={R2} seed {SEED2} ===")
check_trigger()
P, rec2 = stage2_pool(); pool2 = P['records']; keys2 = {key_of(r['p'], r['chi']): r['id'] for r in pool2}
GRIDS = [('NM-none-P5514', 'P5514', 'none', None), ('AB-0.5', 'AB05', '0.5', 0.5)]
OFF = [-60, -30, 0, 30, 60]

# --- (1) grids at R=200 --------------------------------------------------------------------------------------------------------
Z200 = np.random.default_rng(SEED200).standard_normal((R200, K)); ref200 = evaluate_crn(a_u, chi_u, Z200)
assert np.linalg.slogdet(ref200['J'])[1] == json.load(open('mc85_scan_part0.json'))['ldC']
gpts = []
for cid, short, lab, l0 in GRIDS:
    p0 = np.array(rec2[cid]['p']); c0 = np.array(rec2[cid]['chi'])
    for o in itertools.product(OFF, repeat=4):
        gpts.append(dict(id=f"G-{short}({','.join(str(v) for v in o)})", centre=cid, target=lab, offs=list(o), a=np.sqrt(p0).tolist(), chi=(c0+np.array(o)*DEG).tolist()))
t1 = time.time(); raw = eval_many(gpts, Z200)
print(f"{len(gpts)} grid points evaluated at R={R200} in {time.time()-t1:.0f} s")
for c, res in zip(gpts, raw):
    r = summarize_s2(c['id'], c['a'], c['chi'], res, ref200, R200, SEED200)
    c.update({k: r[k] for k in ('B', 'D', 'D_se', 'dec', 'dmD_C', 'mE_C', 'p')}); c['chi'] = r['chi']
    c['f'] = pen(c['dmD_C'], c['mE_C'], c['D'], c['D_se'], None if c['target'] == 'none' else float(c['target']))
grid_start = {}
for cid, short, lab, l0 in GRIDS:
    GP = [c for c in gpts if c['centre'] == cid]; cen = [c for c in GP if c['offs'] == [0, 0, 0, 0]][0]   # (not 'G': that is the plant G(s))
    cnt = {k: sum(c['dec'] == k for c in GP) for k in ('ok', 'over', 'undec')}
    print(f"\nTable G-{short}. Phase grid around {cid} (offsets {OFF} deg on chi1..chi4, powers fixed), R={R200}, seed {SEED200}: decisions {cnt}")
    print(f"  centre (0,0,0,0): dmD_C={cen['dmD_C']:+.3f} mE_C={cen['mE_C']:.3f} B={cen['B']:.2f} D={cen['D']:+.2f}+-{cen['D_se']:.2f} {cen['dec']}")
    print(f"  best qualified (ok) per floor:")
    for lab2, l2 in TARGETS:
        F = [c for c in GP if c['dec'] == 'ok' and (l2 is None or c['mE_C'] >= l2)]
        if F:
            b = max(F, key=lambda c: c['dmD_C'])
            print(f"    lambda0={lab2:4s}: {b['id']:28s} dmD_C={b['dmD_C']:+.3f} (vs centre {b['dmD_C']-cen['dmD_C']:+.3f}) mE_C={b['mE_C']:.3f} B={b['B']:.2f} D={b['D']:+.2f}+-{b['D_se']:.2f}")
        else: print(f"    lambda0={lab2:4s}: no qualified grid point")
    F = [c for c in GP if c['dec'] == 'ok' and (l0 is None or c['mE_C'] >= l0)]
    gs = max(F, key=lambda c: c['dmD_C']) if F else min([c for c in GP if c['dec'] == 'ok'], key=lambda c: c['f'])
    grid_start[cid] = gs
    nb = sum(c['dmD_C'] > cen['dmD_C']+0.05 and c['dec'] == 'ok' and (l0 is None or c['mE_C'] >= l0) for c in GP)
    print(f"  grid start for target {lab}: {gs['id']} (qualified{'' if l0 is None else f', m_E >= {l0}'}); "
          f"qualified grid points{'' if l0 is None else f' with m_E >= {l0}'} more than 0.05 nat above the centre at R={R200}: {nb}")
json.dump(dict(R=R200, seed=SEED200, offsets=OFF, points=gpts), open(f'{OUT}/dense_grids.json', 'w'))

# --- (2) Nelder-Mead starts -------------------------------------------------------------------------------------------------------
LR2 = json.load(open(f'{S2}/local_runs.json')); s2starts = sorted({t[2] for t in LR2['tasks']})
fr2 = json.load(open(f'{S2}/front_R2000.json'))['front']
elig = [r for r in fr2 if r['id'] not in set(s2starts) | {g[0] for g in GRIDS}]
print(f"\nstage-2 starts (excluded): {s2starts}; grid centres (excluded): {[g[0] for g in GRIDS]}")
print(f"eligible stage-2 front points: " + '; '.join(f"{g}: " + ','.join(r['id'] for r in elig if fn(r['mE_C'])) for g, fn in REGIONS))
tasks = []
for cid, short, lab, l0 in GRIDS:
    gs = grid_start[cid]; tasks.append((lab, l0, gs['id'], input_to_x(gs['p'], gs['chi']).tolist()))
    print(f"  target lambda0={lab:4s} grid start {gs['id']} (R={R200}: dmD_C={gs['dmD_C']:+.3f}, mE_C={gs['mE_C']:.3f})")
for lab, l0 in TARGETS:
    chosen = []
    for g, fn in REGIONS:
        cand = [r for r in elig if fn(r['mE_C']) and r['id'] not in chosen]
        if cand: chosen.append(min(cand, key=lambda r: pen_rec(r, l0))['id'])
    rest = sorted([r for r in elig if r['id'] not in chosen], key=lambda r: pen_rec(r, l0))
    while len(chosen) < 4 and rest: chosen.append(rest.pop(0)['id'])
    for i in chosen:
        r = rec2[i]; tasks.append((lab, l0, i, input_to_x(r['p'], r['chi']).tolist()))
        print(f"  target lambda0={lab:4s} start {i:16s} (R=2000: dmD_C={r['dmD_C']:+.3f}, mE_C={r['mE_C']:.3f}, penalised f={pen_rec(r, l0):+.3f})")
setup_search()
t1 = time.time()
with pool(4) as pl: runs = pl.map(nm_run, tasks, chunksize=1)
print(f"\n{len(runs)} Nelder-Mead runs, {sum(r['nfev'] for r in runs)} evaluations at R={RS}, wall {time.time()-t1:.0f} s")
print(f"\nTable N1. Nelder-Mead runs (values at R={RS}, seed {SEEDS}; f = penalised objective of the run's target)")
for r in runs:
    b = r['best_eval']; a, chi = x_to_input(r['x'])
    print(f"  {r['target']:4s} {r['start']:28s} nfev={r['nfev']:3d} f(start)={r['start_eval']['f']:+8.3f} f(end)={r['f']:+8.3f} | end dmD={b['dmD']:+.3f} mE={b['mE']:.3f} "
          f"D={b['D']:+.2f}+-{b['Dse']:.2f} p=({','.join(f'{v:.3f}' for v in a**2)}) chi=({','.join(f'{v:.3f}' for v in chi)})")
json.dump(dict(RS=RS, seedS=SEEDS, MU_B=MU_B, MU_E=MU_E, maxfev=MAXFEV, tasks=tasks, runs=runs), open(f'{OUT}/dense_runs.json', 'w'))

# --- (3) new candidates with provenance, re-evaluated at R=2000 ---------------------------------------------------------------------
new = {}
def origins(x): return [dict(target=r['target'], start=r['start'], eval=i+1) for r in runs for i, v in enumerate(r['archive']) if v['x'] == list(x)]
def add(cid_, a, chi, sel, origin, extra):
    k = key_of(np.asarray(a)**2, chi)
    if k in new: new[k]['selected_by'].append(sel); return
    new[k] = dict(id=cid_, a=list(a), chi=list(chi), selected_by=[sel], origin_run=origin, **extra)
for r in runs:
    a, chi = x_to_input(r['x']); add(f"S3-{r['target']}-{r['start']}", a.tolist(), chi.tolist(), dict(target=r['target'], how='run result'), origins(r['x']), dict(x=r['x'], R500=r['best_eval']))
allv = [v for r in runs for v in r['archive']]
for lab, l0 in TARGETS:
    b = min(allv, key=lambda v: pen(v['dmD'], v['mE'], v['D'], v['Dse'], l0)); a, chi = x_to_input(b['x'])
    add(f"S3AB-{lab}", a.tolist(), chi.tolist(), dict(target=lab, how='archive best (all stage-3 runs)'), origins(b['x']), dict(x=b['x'], R500=b))
for cid, short, lab, l0 in GRIDS:
    for c in sorted([c for c in gpts if c['centre'] == cid], key=lambda c: c['f'])[:20]:
        add(c['id'], c['a'], c['chi'], dict(target=lab, how=f'grid top-20 (R={R200} penalised objective)'), [dict(grid=cid, offsets_deg=c['offs'])],
            dict(R200=dict(dmD=c['dmD_C'], mE=c['mE_C'], D=c['D'], Dse=c['D_se'], dec=c['dec'])))
newl = []
for k, c in new.items():
    if k in keys2: print(f"  {c['id']} coincides with stage-2 pool candidate {keys2[k]} -> not re-evaluated"); continue
    newl.append(c)
Z2 = np.random.default_rng(SEED2).standard_normal((R2, K)); ref = evaluate_crn(a_u, chi_u, Z2)
assert np.linalg.slogdet(ref['J'])[1] == P['ldC'], "R=2000 reference differs from the stage-2 pool reference"
raw = eval_many(newl, Z2); newrecs = []
for c, res in zip(newl, raw):
    r = summarize_s2(c['id'], c['a'], c['chi'], res, ref, R2, SEED2, src='stage3')
    r.update({k: c[k] for k in c if k not in ('id', 'a', 'chi')}); newrecs.append(r)
print(f"\n{len(newrecs)} new candidates re-evaluated at R={R2} (seed {SEED2}; same draws and reference as the stage-2 pool)")
print(f"\nTable N2. New candidates at R={R2}, sorted by dmD_C (top 30 shown; all in dense_R2000.json); provenance: origin_run / selected_by")
print(f"{'id':30s} {'B':>7s} {'D +- se':>14s} {'dec':5s} {'dmD_C':>7s} {'(se)':>7s} {'mE_C':>6s} {'margin':>7s}  origin / selected_by")
def ostr(r): return ', '.join((f"{o['target']}/{o['start']}#{o['eval']}" if 'target' in o else f"grid {o['grid']}{tuple(o['offsets_deg'])}") for o in r['origin_run'])
for r in sorted(newrecs, key=lambda r: -r['dmD_C'])[:30]:
    print(f"{r['id']:30s} {r['B']:7.2f} {r['D']:+7.2f} +- {r['D_se']:4.2f} {r['dec']:5s} {r['dmD_C']:+7.3f} ({r['dmD_se']:5.3f}) {r['mE_C']:6.3f} {-(r['D']+2*r['D_se']):+7.3f}  "
          f"{ostr(r)} / {', '.join(q['target']+' ('+q['how']+')' for q in r['selected_by'])}")
json.dump(dict(R=R2, seed=SEED2, records=newrecs), open(f'{OUT}/dense_R2000.json', 'w'), indent=1)

# --- (4) union front and per-floor comparison ----------------------------------------------------------------------------------------
union = pool2+newrecs; oku = [r for r in union if r['dec'] == 'ok']; fru = [oku[i] for i in pareto_front(oku)]
ok2 = [r for r in pool2 if r['dec'] == 'ok']; fr2ids = {r['id'] for r in fr2}
print(f"\nTable N3. Empirical Pareto front of the evaluated candidates, R={R2} (seed {SEED2}), union of the stage-2 pool ({len(pool2)}) and the stage-3 "
      f"candidates ({len(newrecs)}); qualified {len(oku)}; front {len(fru)} points ('*' = new in stage 3)")
for r in fru:
    print(f"  {'*' if r['id'] not in rec2 else ' '} {r['id']:30s} p={fmt_p(r)} chi={fmt_chi(r)} B={r['B']:7.2f} D={r['D']:+6.2f}+-{r['D_se']:.2f} "
          f"dmD_C={r['dmD_C']:+.3f}({r['dmD_se']:.3f}) mE_C={r['mE_C']:.3f}({r['mE_se']:.4f})")
print(f"  stage-2 front points no longer on the front: {sorted(fr2ids - {r['id'] for r in fru})}")
print(f"\nTable N4. Stage-2 best vs stage-3 (union) best per floor, R={R2}, seed {SEED2}; difference with paired batch se (10x200, same draws)")
print(f"{'lambda0':7s} {'stage-2 best':16s} {'dmD_C':>7s} {'mE_C':>6s} | {'union best':30s} {'dmD_C':>7s} {'mE_C':>6s} {'margin':>7s} | {'difference':>10s} {'(se)':>8s} {'diff/se':>7s}")
cmp_rows = []; ub = []; sb2 = []
for lab, l0 in TARGETS:
    f2 = [r for r in ok2 if l0 is None or r['mE_C'] >= l0-1e-9]; b2 = max(f2, key=lambda r: r['dmD_C'])
    fu = [r for r in oku if l0 is None or r['mE_C'] >= l0-1e-9]; bu = max(fu, key=lambda r: r['dmD_C'])
    d = bu['dmD_C']-b2['dmD_C']; se = diff_se(bu, b2) if bu['id'] != b2['id'] else 0.0
    ub.append(bu['dmD_C']); sb2.append(b2['dmD_C'])
    cmp_rows.append(dict(lambda0=lab, stage2=b2['id'], stage2_dmD=b2['dmD_C'], stage2_mE=b2['mE_C'], union=bu['id'], union_dmD=bu['dmD_C'], union_mE=bu['mE_C'],
                         union_margin=-(bu['D']+2*bu['D_se']), diff=d, diff_se=se))
    print(f"{lab:7s} {b2['id']:16s} {b2['dmD_C']:+7.3f} {b2['mE_C']:6.3f} | {bu['id']:30s} {bu['dmD_C']:+7.3f} {bu['mE_C']:6.3f} {-(bu['D']+2*bu['D_se']):+7.3f} | "
          f"{d:+10.4f} ({se:6.4f}) {(d/se if se > 0 else float('nan')):7.1f}")
print(f"union best per floor none/0.3/0.5/0.7: {np.round(ub, 3).tolist()} -> non-increasing: {all(np.diff(ub) <= 0)}; stage-2: {np.round(sb2, 3).tolist()}")
own = []
for lab, l0 in TARGETS:
    mine = [r for r in newrecs if r['dec'] == 'ok' and (l0 is None or r['mE_C'] >= l0-1e-9) and any(o.get('target') == lab for o in r['origin_run'])]
    sbest = max(mine, key=lambda r: r['dmD_C']) if mine else None; own.append(sbest['dmD_C'] if sbest else np.nan)
    print(f"  own-search outcome lambda0={lab:4s}: " + (f"{sbest['id']} dmD_C={sbest['dmD_C']:+.3f} mE_C={sbest['mE_C']:.3f} [origin {ostr(sbest)}]" if sbest else "none"))
ov = np.array(own); fin = np.isfinite(ov)
print(f"stage-3 own-search outcomes none/0.3/0.5/0.7: {np.round(ov, 3).tolist()} -> non-increasing over the targets with an outcome: {all(np.diff(ov[fin]) <= 0)}")
lam_grid = np.round(np.arange(0, 1.0001, 0.05), 2)
D2 = [max(r['dmD_C'] for r in ok2 if r['mE_C'] >= l-1e-9) for l in lam_grid]; DU = [max(r['dmD_C'] for r in oku if r['mE_C'] >= l-1e-9) for l in lam_grid]
print("max dmD_C over qualified with mE_C >= lambda0, stage-2 pool -> union: " + ', '.join(f"{l:.2f}:{a_:+.3f}->{b_:+.3f}" for l, a_, b_ in zip(lam_grid, D2, DU)))
json.dump(dict(R=R2, seed=SEED2, front=fru, per_floor=cmp_rows, own_search=dict(zip([t[0] for t in TARGETS], own))), open(f'{OUT}/front_union_R2000.json', 'w'), indent=1)

# --- figure: union front --------------------------------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID_, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID_, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 9, 'legend.frameon': False})
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for grp, col, lab_, mk in [([r for r in oku if r['id'] in rec2], '#2a78d6', 'stage-2 pool (qualified)', 'o'), ([r for r in oku if r['id'] not in rec2], '#eb6834', 'stage-3 candidates (qualified)', 's')]:
    ax.plot([r['dmD_C'] for r in grp], [r['mE_C'] for r in grp], mk, ms=4.5, color=col, mec=SURF, mew=0.6, label=lab_, zorder=3)
f2_ = [ok2[i] for i in pareto_front(ok2)]
ax.step([r['dmD_C'] for r in f2_], [r['mE_C'] for r in f2_], where='post', color=MUTED, lw=1.0, ls='--', label='stage-2 empirical front')
ax.step([r['dmD_C'] for r in fru], [r['mE_C'] for r in fru], where='post', color=INK2, lw=1.2, label='union empirical front')
ax.set_xlabel(r'$\Delta m_D^{(C)}$ (nat)'); ax.set_ylabel(r'$m_E^{(C)}$'); ax.set_xlim(left=-0.3)
ax.set_title(f'Empirical Pareto front of the evaluated candidates, R={R2}, seed {SEED2}', loc='left', fontsize=10, color=INK); ax.legend(loc='lower left', fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_front_union.png', dpi=150); plt.close(fig)
log_runtime('stage3_dense', T_START, f'{len(gpts)} grid points at R={R200}, {len(runs)} NM runs, {len(newrecs)} candidates at R={R2}')
