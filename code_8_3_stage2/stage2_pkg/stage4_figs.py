"""§8.5 stage 4: report figures, drawn from the JSON outputs of stage4_search.py, stage4_confirm.py and stage4_scenarios.py (no new
evaluations, no noise drawn). Run from code_8_3_stage2/stage2_pkg/ after the three scripts:  python3 stage4_figs.py
Outputs (stage4/): fig_front.png (front before/after Script 1, zoomed), fig_confirm.png (confirmation intervals), fig_scenarios.png
(representatives across d), stage4_figs.log (the plotted values)."""
import json, os, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = 'stage4'

class Tee:
    def __init__(self, path): self.f = open(path, 'w'); self.o = sys.stdout
    def write(self, m): self.o.write(m); self.f.write(m); self.f.flush()
    def flush(self): self.o.flush(); self.f.flush()
sys.stdout = Tee(f'{OUT}/stage4_figs.log')

INK, INK2, MUTED, GRID_, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'
SER = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']           # categorical slots 1-4 (reference palette), fixed order
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID_, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 9, 'legend.frameon': False})

def front(recs):
    o = sorted(range(len(recs)), key=lambda i: (-recs[i]['dmD_C'], -recs[i]['mE_C'])); f = []; best = -np.inf
    for i in o:
        if recs[i]['mE_C'] > best: f.append(recs[i]); best = recs[i]['mE_C']
    return f

# --- 1. front before / after Script 1 (R=2000, seed 20261012) ---------------------------------------------------------------------
P = json.load(open(f'{OUT}/pool_R2000.json'))['records']; new = {r['id'] for r in json.load(open(f'{OUT}/search_R2000.json'))['records']}
ok = [r for r in P if r['dec'] == 'ok']; old = [r for r in ok if r['id'] not in new]
f0, f1 = front(old), front(ok)
assert [r['id'] for r in f1] == [r['id'] for r in json.load(open(f'{OUT}/front_R2000.json'))['front']], 'stage-4 front mismatch'
assert [r['id'] for r in f0] == [r['id'] for r in json.load(open('stage3/front_union_R2000.json'))['front']], 'stage-3 front mismatch'
print(f"fig_front.png: qualified records {len(ok)} (stage 2-3: {len(old)}, stage 4: {len(ok)-len(old)}); stage-3 front {len(f0)} points, stage-4 front {len(f1)} points")
print("  stage-4 front (dmD_C, mE_C): " + '; '.join(f"{r['id']} ({r['dmD_C']:+.3f}, {r['mE_C']:.3f})" for r in f1))
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for grp, col, lab, mk in [(old, SER[0], 'stage 2-3 (qualified)', 'o'), ([r for r in ok if r['id'] in new], SER[1], 'stage 4 (qualified)', 's')]:
    ax.plot([r['dmD_C'] for r in grp], [r['mE_C'] for r in grp], mk, ms=4.5, color=col, mec=SURF, mew=0.6, label=lab, zorder=3)
ax.step([r['dmD_C'] for r in f0], [r['mE_C'] for r in f0], where='post', color=MUTED, lw=1.0, ls='--', label='stage-3 empirical front', zorder=2)
ax.step([r['dmD_C'] for r in f1], [r['mE_C'] for r in f1], where='post', color=INK2, lw=1.3, label='stage-4 empirical front', zorder=2)
for l0 in (0.3, 0.5, 0.7):
    ax.axhline(l0, color=MUTED, lw=0.7, ls=':', zorder=1); ax.text(-0.29, l0+0.008, f'λ0 = {l0}', fontsize=7.5, color=MUTED)
ax.set_xlim(-0.3, 0.8); ax.set_ylim(-0.02, 1.04)
ax.set_xlabel(r'$\Delta m_D^{(C)}$ (nat)'); ax.set_ylabel(r'$m_E^{(C)}$')
ax.set_title('Empirical Pareto front of the evaluated candidates, R=2000, seed 20261012 (x clipped at −0.3)', loc='left', fontsize=9.5, color=INK)
ax.legend(loc='upper right', fontsize=8); fig.tight_layout(); fig.savefig(f'{OUT}/fig_front.png', dpi=150); plt.close(fig)

# --- 2. confirmation: corrected intervals (seed 20261020, R=8000) --------------------------------------------------------------------
CF = json.load(open(f'{OUT}/confirm_R8000.json')); C = CF['candidates']; S = CF['sets']; tc = CF['frozen']['t_crit']
member = {c['id']: [lab for lab in S if c['id'] in S[lab]['near_opt_confirmed']+S[lab]['near_opt_undecided']] for c in C}
print(f"\nfig_confirm.png: {len(C)} candidates, Delta m_D^(C) and m_E^(C) with intervals +- {tc:.4f} se (seed 20261020, R=8000); near-optimal membership per floor:")
for c in sorted(C, key=lambda c: -c['dmD']):
    print(f"  {c['id']:36s} dmD [{c['dmD_lo']:+.4f},{c['dmD_hi']:+.4f}] mE [{c['mE_lo']:.4f},{c['mE_hi']:.4f}] member of: {', '.join(member[c['id']]) or '-'}")
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for c in C:
    inset = bool(member[c['id']]); col = SER[1] if inset else SER[0]
    ax.plot([c['dmD_lo'], c['dmD_hi']], [c['mE'], c['mE']], color=col, lw=1.6, solid_capstyle='round', zorder=3)
    ax.plot([c['dmD']], [c['mE']], 'o', ms=4.5, color=col, mec=SURF, mew=0.6, zorder=4)
ax.plot([], [], 'o-', color=SER[1], ms=4.5, label='in a candidate near-optimal set (some floor)')
ax.plot([], [], 'o-', color=SER[0], ms=4.5, label='not in any candidate near-optimal set')
for l0 in (0.3, 0.5, 0.7):
    ax.axhline(l0, color=MUTED, lw=0.7, ls=':', zorder=1); ax.text(-0.04, l0+0.008, f'λ0 = {l0}', fontsize=7.5, color=MUTED)
ax.set_xlim(-0.05, 0.78); ax.set_ylim(-0.02, 1.04); ax.set_xlabel(r'$\Delta m_D^{(C)}$ (nat), corrected interval'); ax.set_ylabel(r'$m_E^{(C)}$ (point estimate)')
ax.set_title(f'Frozen pool, confirmation batch seed 20261020, R=8000; intervals ± {tc:.2f} se (M = {CF["frozen"]["family"]["M"]})', loc='left', fontsize=9.5, color=INK)
ax.legend(loc='upper right', fontsize=8); fig.tight_layout(); fig.savefig(f'{OUT}/fig_confirm.png', dpi=150); plt.close(fig)

# --- 3. representatives across d (seed 20261021, R=4000, same draws) ------------------------------------------------------------------
SC = json.load(open(f'{OUT}/scenarios.json')); ds = [5, 20, 50]
rows = {d: {q['id']: q for q in SC['scenarios'][str(d)]['rows']} for d in ds}
reps = [(lab, v['id']) for lab, v in SC['representatives'].items() if v]
print(f"\nfig_scenarios.png: representatives at d = 5, 20 (control), 50; seed 20261021, R=4000")
for lab, i in reps:
    print(f"  {lab:4s} {i:36s} dmD_C " + ' '.join(f"d={d}: {rows[d][i]['dmD_C']:+.3f}" for d in ds) + " | D " + ' '.join(f"d={d}: {rows[d][i]['D']:+.3f}+-{rows[d][i]['D_se']:.3f}" for d in ds))
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9))
for k, (lab, i) in enumerate(reps):
    for ax_, key in ((axs[0], 'dmD_C'), (axs[1], 'D')):
        y = [rows[d][i][key] for d in ds]; e = [rows[d][i]['dmD_se' if key == 'dmD_C' else 'D_se'] for d in ds]
        ax_.errorbar(range(3), y, yerr=[2*v for v in e], color=SER[k], lw=2, marker='o', ms=5, mec=SURF, capsize=0, label=f'λ0 = {lab}' if lab != 'none' else 'λ0 = none')
for ax_ in axs: ax_.set_xticks(range(3)); ax_.set_xticklabels(['d = 5', 'd = 20 (control)', 'd = 50']); ax_.set_xlim(-0.2, 2.2)
axs[0].axhline(0, color=MUTED, lw=0.8); axs[1].axhline(0, color=MUTED, lw=0.8)
axs[0].set_ylabel(r'$\Delta m_D^{(C)}$ vs the scenario uniform reference (nat)'); axs[1].set_ylabel('paired send difference D (sends)')
axs[0].set_title('D-criterion gain (±2 se)', loc='left', fontsize=9.5, color=INK); axs[1].set_title('budget: D < 0 qualifies (±2 se)', loc='left', fontsize=9.5, color=INK)
axs[0].legend(loc='lower right', fontsize=7.5); fig.tight_layout(); fig.savefig(f'{OUT}/fig_scenarios.png', dpi=150); plt.close(fig)
print("figures saved")

# --- derived checks for the predictions table of stage4_report.md (from stage4/front_R2000.json; no new numbers) ------------------
F = json.load(open(f'{OUT}/front_R2000.json')); se_ = F['start_end']
print("\nDerived checks (Script 1, R=2000 seed 20261012, from front_R2000.json 'start_end'):")
for lab in ('0.3', '0.5', '0.7'):
    g = [q for q in se_ if q['run'].startswith(lab+'/') and q['end_feasible']]
    print(f"  P1.3 floor {lab}: {len(g)} feasible endpoints; end m_E - floor = {[round(q['end_mE']-float(lab), 3) for q in g]}; within 0.03: {sum(q['end_mE']-float(lab) <= 0.03 for q in g)}")
fr_ = [q for q in se_ if q['src'] == 'front' and q['start_feasible']]; ch_ = np.array([q['change'] for q in fr_])
print(f"  P1.4 feasible front starts: n={len(fr_)}, change median {np.median(ch_):+.4f}, |change| <= 0.03: {int(np.sum(np.abs(ch_) <= 0.03))}, "
      f"larger: " + ', '.join(f"{q['run']} {q['change']:+.4f} (m_E {q['start_mE']:.3f} -> {q['end_mE']:.3f})" for q in fr_ if abs(q['change']) > 0.03))
for lab in ('none', '0.3', '0.5', '0.7'):
    g = [q for q in se_ if q['run'].startswith(lab+'/') and q['src'] == 'R200']
    print(f"  P1.4 R=200 pool starts, floor {lab}: start feasible {sum(q['start_feasible'] for q in g)}/{len(g)}, end feasible {sum(q['end_feasible'] for q in g)}/{len(g)}; "
          f"start failures {sorted(set(q['start_fails'] for q in g)) or ['none']}; changes of feasible starts "
          + (', '.join(f"{q['change']:+.3f}" for q in g if q['start_feasible']) or '-') + "; best feasible end " + (f"{max(q['end_dmD'] for q in g if q['end_feasible']):+.3f}" if any(q['end_feasible'] for q in g) else '-'))
byid_ = {r['id']: r for r in P}
print("\nDerived check: new stage-4 front points vs the start of the run that produced them (R=2000, seed 20261012):")
for r in f1:
    if r['id'] not in new: continue
    w = r['kept_as'][0]; st = byid_[w['run'].split('/', 1)[1]]
    dp = np.max(np.abs(np.array(r['p'])-np.array(st['p']))); dc = np.max(np.abs((np.array(r['chi'])-np.array(st['chi'])+np.pi) % (2*np.pi)-np.pi))
    print(f"  {r['id']:36s} from run {w['run']} ({w['kind']} #{w['eval']}): max|dp| = {dp:.1e}, max|dchi| = {dc:.1e} rad, "
          f"dmD_C {st['dmD_C']:+.4f} -> {r['dmD_C']:+.4f} ({r['dmD_C']-st['dmD_C']:+.4f}), mE_C {st['mE_C']:.3f} -> {r['mE_C']:.3f}")
