"""§8.5 stage 3, experiment 1: how sharp are the current points? One-dimensional sweeps at R=200 (seed 20261010, the screening
draws; reference (C) = uniform on the same draws) around NM-none-P5514 and AB-0.5 (p, chi from stage2/pool_R2000.json):
  (a) chi3 alone, (b) chi4 alone, (c) chi3 - chi4 with chi3 + chi4 fixed (chi3 + d/2, chi4 - d/2), each d in [-60, 60] deg, step 5;
  (d) p1 from 0.15 to 0.40 in 0.01 steps, the other three powers rescaled proportionally;
  (e) chi1 alone and chi2 alone, d in [-60, 60] deg, step 10.
Per point: Delta m_D^(C), m_E^(C), B, D +- se and the 8.3.6(g) decision (qual, mc85_post.py).
Half-width: distance from the centre (the stage-2 point) to where Delta m_D first falls 0.05 nat below its centre value, on each side,
by linear interpolation between sweep points; 'none' = no such crossing within the sweep range. Each side is also labelled with the
8.3.6(g) decision of its first sweep point; the within-budget half-width is the smaller half-width over the sides whose first point is
not 'over' (the side that leaves the budget is not a feasible direction).
Control (an addition, not in the request): the seed-20261010 R=200 block is the first 200 rows of the R=500 block on which the stage-2
Nelder-Mead searched, so both centres are selection-biased on it. The same sweeps are therefore repeated on a fresh, previously unused
block (seed 20261017, R=200, its own uniform reference); the requested seed-20261010 results are reported first and unchanged.
Run from code_8_3_stage2/stage2_pkg/:  python3 stage3_peak.py      Outputs (stage3/): stage3_peak.log, peak.json, fig_peak_dmD.png, fig_peak_D.png, fig_peak_dmD_control.png"""
exec(open('stage3_common.py').read())
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
T_START = time.time(); sys.stdout = Tee(f'{OUT}/stage3_peak.log')
print(f"=== stage3_peak.py: 1-D sweeps at R={R200}, seed {SEED200} (screening draws), reference (C) uniform on the same draws ===")
check_trigger()
P, rec = stage2_pool()
SEED_CTRL = 20261017                       # control block, never used before in this project
BLOCKS = [('main', SEED200), ('control', SEED_CTRL)]
Zb = {lab: np.random.default_rng(sd).standard_normal((R200, K)) for lab, sd in BLOCKS}
refb = {lab: evaluate_crn(a_u, chi_u, Zb[lab]) for lab, _ in BLOCKS}
ldC200 = np.linalg.slogdet(refb['main']['J'])[1]; ld_s1 = json.load(open('mc85_scan_part0.json'))['ldC']
assert ldC200 == ld_s1, "R=200 reference differs from the stage-1/LHS screening reference"
print(f"reference (C), main block seed {SEED200}: B_u={refb['main']['cnt'].mean():.2f}, log det J_C={ldC200:.6f} (identical to the stage-1/LHS screening reference)")
print(f"reference (C), control block seed {SEED_CTRL}: B_u={refb['control']['cnt'].mean():.2f}, log det J_C={np.linalg.slogdet(refb['control']['J'])[1]:.6f}")
CENTRES = ['NM-none-P5514', 'AB-0.5']
PH = np.arange(-60, 61, 5); PH10 = np.arange(-60, 61, 10); P1 = np.round(np.arange(15, 41)/100, 2)
SWEEPS = [('a', 'chi3 alone', 'deg'), ('b', 'chi4 alone', 'deg'), ('c', 'chi3-chi4, chi3+chi4 fixed', 'deg'),
          ('d', 'p1 (others rescaled)', 'p1'), ('e1', 'chi1 alone', 'deg'), ('e2', 'chi2 alone', 'deg')]
pts = []
for cid in CENTRES:
    p0 = np.array(rec[cid]['p']); c0 = np.array(rec[cid]['chi'])
    pts.append(dict(centre=cid, sweep='centre', x=0.0, a=np.sqrt(p0).tolist(), chi=c0.tolist()))
    for d in PH:
        for sw, dc in [('a', [0, 0, d, 0]), ('b', [0, 0, 0, d]), ('c', [0, 0, d/2, -d/2])]:
            pts.append(dict(centre=cid, sweep=sw, x=float(d), a=np.sqrt(p0).tolist(), chi=(c0+np.array(dc)*DEG).tolist()))
    for d in PH10:
        for sw, dc in [('e1', [d, 0, 0, 0]), ('e2', [0, d, 0, 0])]:
            pts.append(dict(centre=cid, sweep=sw, x=float(d), a=np.sqrt(p0).tolist(), chi=(c0+np.array(dc)*DEG).tolist()))
    for q in P1:
        p = np.concatenate([[q], p0[1:]*(1-q)/(1-p0[0])]); assert abs(p.sum()-1) < 1e-12
        pts.append(dict(centre=cid, sweep='d', x=float(q), a=np.sqrt(p).tolist(), chi=c0.tolist()))
base = pts; pts = []
for lab, sd in BLOCKS:
    t1 = time.time(); raw = eval_many(base, Zb[lab])
    print(f"{len(base)} sweep points evaluated at R={R200} on the {lab} block (seed {sd}) in {time.time()-t1:.0f} s (4 workers)")
    for c0_, res in zip(base, raw):
        c = dict(c0_, block=lab, seed=sd)
        r = summarize_s2(f"{c['centre']}:{c['sweep']}:{c['x']:g}", c['a'], c['chi'], res, refb[lab], R200, sd)
        c.update({k: r[k] for k in ('B', 'D', 'D_se', 'dec', 'dmD_C', 'mE_C', 'p', 'lamC')}); c['chi'] = r['chi']; pts.append(c)

def halfwidth(xs, ys, x0, y0, drop=0.05):
    L = y0-drop; out = {}
    for side, sel in (('left', xs < x0), ('right', xs > x0)):
        o = np.argsort(np.abs(xs[sel]-x0)); xx = xs[sel][o]; yy = ys[sel][o]; px, py = x0, y0; hw = None
        for x_, y_ in zip(xx, yy):
            if y_ < L: hw = float(abs(px+(L-py)*(x_-px)/(y_-py)-x0)); break
            px, py = x_, y_
        out[side] = hw
    return out
summ = []
for blk, sd in BLOCKS:
  for cid in CENTRES:
    cen = [c for c in pts if c['block'] == blk and c['centre'] == cid and c['sweep'] == 'centre'][0]
    print(f"\n######## [{blk} block, seed {sd}] centre {cid}: p={fmt_p(cen)} chi={fmt_chi(cen)}; at R={R200}: dmD_C={cen['dmD_C']:+.3f} mE_C={cen['mE_C']:.3f} "
          f"B={cen['B']:.2f} D={cen['D']:+.2f}+-{cen['D_se']:.2f} {cen['dec']}  (R=2000, seed 20261012: dmD_C={rec[cid]['dmD_C']:+.3f})")
    for sw, lab, unit in SWEEPS:
        S = sorted([c for c in pts if c['block'] == blk and c['centre'] == cid and c['sweep'] == sw], key=lambda c: c['x'])
        xs = np.array([c['x'] for c in S]); ys = np.array([c['dmD_C'] for c in S])
        x0 = cen['p'][0] if sw == 'd' else 0.0
        print(f"\nTable P-{blk}-{cid}-{sw}. Sweep ({sw}) {lab}; x = {'p1' if sw == 'd' else 'offset in degrees'}; R={R200}, seed {sd} ({blk} block)")
        print(f"{'x':>7s} {'dmD_C':>8s} {'d-centre':>8s} {'mE_C':>6s} {'B':>8s} {'D +- se':>14s} {'dec':5s}")
        for c in S:
            print(f"{c['x']:7g} {c['dmD_C']:+8.3f} {c['dmD_C']-cen['dmD_C']:+8.3f} {c['mE_C']:6.3f} {c['B']:8.2f} {c['D']:+7.2f} +- {c['D_se']:4.2f} {c['dec']:5s}")
        hw = halfwidth(xs, ys, x0, cen['dmD_C']); step = 0.1 if sw == 'd' else 60.0
        lft = [c for c in S if c['x'] < x0]; rgt = [c for c in S if c['x'] > x0]
        dl = lft[-1]['dec'] if lft else None; dr = rgt[0]['dec'] if rgt else None
        feas = [v for v, d_ in ((hw['left'], dl), (hw['right'], dr)) if d_ != 'over' and v is not None]
        nofeas_cross = [d_ for v, d_ in ((hw['left'], dl), (hw['right'], dr)) if d_ != 'over' and v is None]
        hw_in = min(feas) if feas else None
        im = int(np.argmax(ys)); okx = [c['x'] for c in S if c['dec'] == 'ok']
        fmt = (lambda v: 'none' if v is None else f"{v:.4f}") if sw == 'd' else (lambda v: 'none' if v is None else f"{v:.1f}")
        print(f"half-width at centre-0.05 nat: left {fmt(hw['left'])} [first point {dl}], right {fmt(hw['right'])} [first point {dr}] ({'p1 units' if sw == 'd' else 'deg'}); "
              f"within-budget half-width {fmt(hw_in)}{' (a within-budget side has no crossing)' if nofeas_cross else ''}; "
              f"stage-1 grid step {step:g}; sweep max dmD_C {ys[im]:+.3f} at x={xs[im]:g} ({ys[im]-cen['dmD_C']:+.3f} vs centre); "
              f"qualified (ok) at R=200 for x in {('[' + f'{min(okx):g}, {max(okx):g}' + ']') if okx else 'none'} ({len(okx)}/{len(S)} points)")
        summ.append(dict(block=blk, seed=sd, centre=cid, sweep=sw, label=lab, left=hw['left'], right=hw['right'], left_dec=dl, right_dec=dr,
                         within_budget=hw_in, within_budget_side_without_crossing=bool(nofeas_cross), grid_step=step, centre_dmD=cen['dmD_C'],
                         sweep_max=float(ys[im]), x_max=float(xs[im]), n_ok=len(okx), ok_range=[min(okx), max(okx)] if okx else None))
for blk, sd in BLOCKS:
    print(f"\nTable P-summary-{blk}. Half-widths at centre - 0.05 nat (R={R200}, seed {sd}, {blk} block); phase sweeps in degrees (stage-1 step 60 deg), "
          "p1 sweep in p1 units (stage-1 steps 0.1 coarse / 0.05 near-uniform); 'none' = no crossing within the sweep range; [ok/undec/over] = decision "
          "of the first point on that side; 'in-budget' = smaller half-width over the sides whose first point is not over budget")
    print(f"{'centre':14s} {'sweep':5s} {'what':28s} {'left':>8s} {'[dec]':7s} {'right':>8s} {'[dec]':7s} {'in-budget':>9s} {'/step':>6s} {'sweep max - centre':>18s} {'at x':>6s} {'dec':5s}")
    for s_ in [q for q in summ if q['block'] == blk]:
        f_ = (lambda v: '    none' if v is None else (f"{v:8.4f}" if s_['sweep'] == 'd' else f"{v:8.1f}"))
        ratio = None if s_['within_budget'] is None else s_['within_budget']/s_['grid_step']
        dmx = [c['dec'] for c in pts if c['block'] == blk and c['centre'] == s_['centre'] and c['sweep'] == s_['sweep'] and c['x'] == s_['x_max']][0]
        print(f"{s_['centre']:14s} {s_['sweep']:5s} {s_['label']:28s} {f_(s_['left'])} [{str(s_['left_dec']):5s}] {f_(s_['right'])} [{str(s_['right_dec']):5s}] "
              f"{f_(s_['within_budget']):>9s} {('  none' if ratio is None else f'{ratio:6.3f}')} {s_['sweep_max']-s_['centre_dmD']:+18.3f} {s_['x_max']:6g} {dmx:5s}")
json.dump(dict(R=R200, seeds=dict(BLOCKS), ldC_main=ldC200, points=pts, summary=summ), open(f'{OUT}/peak.json', 'w'), indent=1)

# --- figures: one panel per sweep, both centres -----------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'; COL = {'NM-none-P5514': '#2a78d6', 'AB-0.5': '#eb6834'}
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 9, 'axes.titlesize': 9.5, 'axes.titlecolor': INK, 'legend.frameon': False})
for key, ylab, fname, blk in [('dmD_C', r'$\Delta m_D^{(C)}$ − centre value (nat)', 'fig_peak_dmD.png', 'main'), ('D', 'D = paired send difference vs uniform', 'fig_peak_D.png', 'main'),
                              ('dmD_C', r'$\Delta m_D^{(C)}$ − centre value (nat)', 'fig_peak_dmD_control.png', 'control')]:
    sd = dict(BLOCKS)[blk]
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 6.4))
    for ax, (sw, lab, unit) in zip(axs.ravel(), SWEEPS):
        for cid in CENTRES:
            cen = [c for c in pts if c['block'] == blk and c['centre'] == cid and c['sweep'] == 'centre'][0]
            S = sorted([c for c in pts if c['block'] == blk and c['centre'] == cid and c['sweep'] == sw], key=lambda c: c['x'])
            xs = [c['x'] for c in S]; ys = [c[key]-(cen[key] if key == 'dmD_C' else 0) for c in S]
            ax.plot(xs, ys, '-o', ms=3, lw=1.2, color=COL[cid], label=cid)
            if sw == 'd': ax.axvline(cen['p'][0], color=COL[cid], lw=0.8, ls=':')
        if key == 'dmD_C': ax.axhline(-0.05, color=INK2, lw=0.8, ls='--')
        else: ax.axhline(0, color=INK2, lw=0.8, ls='--')
        ax.set_title(f"({sw}) {lab}", loc='left'); ax.set_xlabel('p1' if sw == 'd' else 'offset from centre (deg)')
    axs[0, 0].set_ylabel(ylab); axs[1, 0].set_ylabel(ylab); axs[0, 0].legend(fontsize=8, loc='lower center')
    fig.suptitle(f"1-D sweeps around the stage-2 points, R={R200}, seed {sd} ({blk} block)" + (" (dashed: centre − 0.05 nat)" if key == 'dmD_C' else " (dashed: D = 0; dotted: centre p1)"),
                 x=0.01, ha='left', color=INK, fontsize=10)
    fig.tight_layout(); fig.savefig(f'{OUT}/{fname}', dpi=150); plt.close(fig)
print("figures saved: fig_peak_dmD.png, fig_peak_D.png (main block), fig_peak_dmD_control.png (control block)")
log_runtime('stage3_peak', T_START, f'{len(pts)} sweep evaluations at R={R200} (main + control block)')
