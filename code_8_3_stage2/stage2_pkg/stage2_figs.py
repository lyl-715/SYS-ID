"""§8.5 stage 2, figures for stage2_report.md (reads the JSON/NPZ outputs of stage2_post/refine/local/final).
Run from code_8_3_stage2/stage2_pkg/:  python3 stage2_figs.py
Outputs (stage2/): fig_pareto_R2000.png, fig_dmD_vs_B_R200.png, fig_qxw.png, stage2_figs.log"""
import json, sys, time, numpy as np
T_START = time.time()
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
OUT = 'stage2'
# reference palette (light): text inks, recessive grid, categorical slots 1-3, blue sequential ramp
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'
BLUE = LinearSegmentedColormap.from_list('blue', ['#cde2fb', '#86b6ef', '#3987e5', '#2a78d6', '#1c5cab', '#104281', '#0d366b'])
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9, 'axes.titlesize': 10,
                     'axes.titlecolor': INK, 'legend.frameon': False, 'lines.linewidth': 1.0})
log = open(f'{OUT}/stage2_figs.log', 'w')
def say(m): print(m); log.write(m+'\n')

# --- 1. empirical Pareto front at R=2000 ------------------------------------------------------------------------------
P = json.load(open(f'{OUT}/pool_R2000.json')); recs = P['records']; fr = json.load(open(f'{OUT}/front_R2000.json'))['front']
fin = {r['id']: r['roles'] for r in json.load(open(f'{OUT}/final_batch1.json'))['records']}
ok = [r for r in recs if r['dec'] == 'ok']
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for grp, col, lab, mk in [([r for r in ok if r['src'] not in ('nm', 'ctrl')], S1, 'refined candidates (stage2_refine)', 'o'),
                          ([r for r in ok if r['src'] == 'nm'], S2, 'Nelder-Mead results (stage2_local)', 's'),
                          ([r for r in ok if r['src'] == 'ctrl'], S3, 'phase-only controls (stage2_local)', 'D')]:
    ax.errorbar([r['dmD_C'] for r in grp], [r['mE_C'] for r in grp], xerr=[2*r['dmD_se'] for r in grp], fmt=mk, ms=5, color=col,
                mec=SURF, mew=0.8, elinewidth=0.8, capsize=0, label=lab, zorder=3)
fx = [r['dmD_C'] for r in fr]; fy = [r['mE_C'] for r in fr]
ax.step(fx, fy, where='post', color=INK2, lw=1.0, zorder=2, label='empirical Pareto front (qualified, R=2000)')
ax.plot(fx, fy, 'o', ms=9, mfc='none', mec=INK2, mew=0.8, zorder=2)
for r in fr:
    if r['id'] in fin:
        ax.annotate(r['id'], (r['dmD_C'], r['mE_C']), xytext=(6, 4), textcoords='offset points', fontsize=7.5, color=INK)
for i, roles in fin.items():
    if i not in [r['id'] for r in fr]:
        r = [q for q in recs if q['id'] == i][0]; ax.annotate(i, (r['dmD_C'], r['mE_C']), xytext=(6, -10), textcoords='offset points', fontsize=7.5, color=INK2)
ax.set_xlabel(r'$\Delta m_D^{(C)}$  (nat, vs uniform on the same draws)'); ax.set_ylabel(r'$m_E^{(C)}$')
ax.set_title(f'Empirical Pareto front of the evaluated candidates, R={P["R"]}, seed {P["seed"]}', loc='left')
ax.text(0.01, 0.99, 'qualified candidates only; bars: $\\pm$2 paired batch se of $\\Delta m_D^{(C)}$', transform=ax.transAxes, fontsize=7.5, color=INK2, va='top')
ax.legend(loc='lower left', fontsize=8); fig.tight_layout(); fig.savefig(f'{OUT}/fig_pareto_R2000.png', dpi=150); plt.close(fig)
say(f"fig_pareto_R2000.png: {len(ok)} qualified points at R=2000 ({sum(r['src']=='nm' for r in ok)} from Nelder-Mead, {sum(r['src']=='ctrl' for r in ok)} phase-only controls), front {len(fr)} points")

# --- 2. Delta m_D vs B at R=200 -------------------------------------------------------------------------------------------
M = json.load(open(f'{OUT}/merged_R200.json'))['records']
B = np.array([r['B'] for r in M]); Dm = np.array([r['dmD_C'] for r in M]); mE = np.array([r['mE_C'] for r in M])
isl = np.array([r['src'] == 'lhs' for r in M]); Bu = [r['B'] for r in M if r['id'] == 'UNIF'][0]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
for ax, zoom in zip(axs, [False, True]):
    sel = np.ones(len(M), bool) if not zoom else (np.abs(B-Bu) <= 40) & (Dm >= -1.5)
    for m_, mk, lab in [(isl, 'o', 'LHS (amplitude + phase)'), (~isl, '^', 'stage-1 grids')]:
        k = sel & m_
        sc = ax.scatter(B[k], Dm[k], c=mE[k], cmap=BLUE, vmin=0, vmax=1, s=9 if zoom else 6, marker=mk, linewidths=0.25, edgecolors=SURF, label=lab)
    ax.axvline(Bu, color=INK2, lw=0.8, ls='--'); ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel(r'expected sends $\hat B_u$'); ax.set_ylabel(r'$\Delta m_D^{(C)}$ (nat)')
    ax.set_title(('all 4458 candidates, R=200, seed 20261010' if not zoom else f'zoom: |B - B_unif| <= 40, dmD >= -1.5 ({sel.sum()} points)'), loc='left')
    if not zoom: ax.annotate(f'uniform budget {Bu:.1f}', (Bu, Dm.min()), xytext=(4, 4), textcoords='offset points', fontsize=7.5, color=INK2)
axs[1].legend(loc='lower right', fontsize=8, markerscale=1.6)
cb = fig.colorbar(sc, ax=axs[1], pad=0.02); cb.set_label(r'$m_E^{(C)}$', color=INK2); cb.outline.set_visible(False)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_dmD_vs_B_R200.png', dpi=150); plt.close(fig)
say(f"fig_dmD_vs_B_R200.png: {len(M)} candidates (LHS {isl.sum()}), budget line at B_unif={Bu:.2f}")

# --- 3. q_k, x_k, w_k for three representatives ---------------------------------------------------------------------------------
Q = np.load(f'{OUT}/final_qxw.npz'); t = Q['t']
b1 = json.load(open(f'{OUT}/final_batch1.json')); best = P['best']
reps = [best['none'], next((i for i in (best['0.5'], best['0.3']) if i not in (best['none'], 'UNIF')), best['0.3']), 'UNIF']
names = {reps[0]: 'best unconstrained', reps[1]: r'best with $m_E \geq 0.5$' if reps[1] == best['0.5'] else r'best with $m_E \geq 0.3$', 'UNIF': 'uniform reference'}
fig, axs = plt.subplots(3, 3, figsize=(11, 6.6), sharex=True, sharey='row')
for j, i in enumerate(reps):
    rr = [r for r in b1['records'] if r['id'] == i][0]
    axs[0, j].plot(t, Q[f'{i}_x'], color=S1, lw=0.8); axs[1, j].plot(t[1:], Q[f'{i}_q'][1:], color=S2, lw=0.6); axs[2, j].plot(t, Q[f'{i}_w'], color=S3, lw=0.6)
    axs[0, j].set_title(f"{i}: {names[i]}\n$\\Delta m_D^{{(C)}}$={rr['dmD_C']:+.3f}, $m_E^{{(C)}}$={rr['mE_C']:.3f}, B={rr['B']:.1f}", loc='left', fontsize=9)
    axs[2, j].set_xlabel('t (s)')
    say(f"fig_qxw.png panel {i}: q mean {Q[f'{i}_q'].mean():.4f} max over k>=1 {Q[f'{i}_q'][1:].max():.3f}; w mean {Q[f'{i}_w'].mean():.4f} min {Q[f'{i}_w'].min():.4f}")
axs[0, 0].set_ylabel('$x_k$ (noise-free output)'); axs[1, 0].set_ylabel('$q_k$ (event rate, k>=1)'); axs[2, 0].set_ylabel('$w_k$ (information weight)')
fig.suptitle(f"Event rate $q_k$, output $x_k$ and weight $w_k$ per clock point, batch 1 (R={b1['R']}, seed {b1['seed']})", x=0.01, ha='left', color=INK, fontsize=10)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_qxw.png', dpi=150); plt.close(fig)
el = time.time()-T_START
with open(f'{OUT}/runtime.log', 'a') as f: f.write(f"stage2_figs\tstarted {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(T_START))}\telapsed {el:.1f} s ({el/60:.2f} min)\t3 figures\n")
log.close()
