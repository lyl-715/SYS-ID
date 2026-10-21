"""Round 2c figures (no number in the report depends on them; every plotted value is read from results/ or logs/).
  gamma   fig_gamma.png: gamma^(1)_k and gamma^(2)_k over t_k (results/gamma_A1.npy, gamma_A2.npy)
  dual    fig_dual_lambda.png: D(A_i, lambda) - 197.197 at the evaluated lambda of the golden sections (results/outer_*.json),
          two panels (2-D, 1-D), with the S-route value (results/part_c2.json)
Usage (from this directory): python3 plots.py gamma | dual"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'results')
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#8a8984', '#fcfcfb'
COL = {1: '#2a78d6', 2: '#eb6834'}                        # categorical slots 1, 2 (blue, orange)
LAB = {1: 'A₁ = M(w(P5514))⁻¹', 2: 'A₂ = M(w_U)⁻¹'}
LD_JC = 197.197


def style(ax):
    ax.set_facecolor(SURF); ax.tick_params(labelsize=8, colors=INK2)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(MUTED)
    ax.grid(alpha=0.25, lw=0.5)

def fig_gamma():
    dt = 1e-3; fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True); fig.patch.set_facecolor(SURF)
    for i in (1, 2):
        gam = np.load(os.path.join(RES, f'gamma_A{i}.npy')); t = np.arange(1, gam.size+1)*dt
        ax.plot(t, gam, color=COL[i], lw=1.2, label=f'{LAB[i]}  (mean {gam.mean():.5f})')
        ax.axhline(gam.mean(), color=COL[i], lw=0.8, ls=':')
        ax.annotate(LAB[i], (t[-1], gam[-1]), xytext=(6, 0), textcoords='offset points', fontsize=8, color=INK2, va='center')
    ax.set_xlabel('t_k (s)', fontsize=9, color=INK2); ax.set_ylabel('γ_k = σ⁻² h_U,kᵀ A h_U,k', fontsize=9, color=INK2)
    ax.set_xlim(0, 5.6); style(ax); ax.legend(fontsize=8, frameon=False, loc='upper center', ncol=2)
    ax.set_title('Per-step coefficients γ_k of the two fixed A (dotted: mean over k); Σ_k w_k γ_k = 9 at the weights that define A',
                 fontsize=10, color=INK, loc='left')
    fig.savefig(os.path.join(HERE, 'fig_gamma.png'), dpi=130, facecolor=SURF); plt.close(fig)

def fig_dual():
    s = json.load(open(os.path.join(RES, 'part_c2.json'))); sv = s['value']
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True); fig.patch.set_facecolor(SURF)
    for ax, mod, ttl in zip(axs, ('2d', '1d'), ('2-D model, acceleration constraint', '1-D model, no acceleration constraint')):
        lo_all = []
        for i in (1, 2):
            r = json.load(open(os.path.join(RES, f'outer_{mod}_A{i}.json'))); gb = r['gamma_bar']
            lam = np.array([float(k) for k in r['evals']]); D = np.array([v['D'] for v in r['evals'].values()])
            o = np.argsort(lam); x = lam[o]/gb; y = D[o]-LD_JC; lo_all.append(y.min())
            ax.plot(x, y, color=COL[i], lw=1.5, marker='o', ms=4, mec=SURF, mew=1.0, label=LAB[i])
            ax.plot([r['lam_star_over_gamma_bar']], [r['D_min_minus_LDJC']], marker='o', ms=9, mfc='none', mec=COL[i], mew=1.5)
            ax.annotate(f"min {r['D_min_minus_LDJC']:.3f} at λ*/γ_bar = {r['lam_star_over_gamma_bar']:.3f}", (r['lam_star_over_gamma_bar'], r['D_min_minus_LDJC']),
                        xytext=(28, -4 if i == 1 else 6), textcoords='offset points', fontsize=8, color=INK2,
                        arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
        ax.axhline(sv, color=INK2, lw=1.0, ls='--')
        ax.annotate(f'S route: U(619.129) − 197.197 = {sv:.3f}', (4.9, sv), xytext=(0, 4), textcoords='offset points', fontsize=8, color=INK2, ha='right')
        ax.set_xlim(0, 5); ax.set_ylim(min(sv, min(lo_all))-0.6, max(lo_all)+6.0); ax.set_title(ttl, fontsize=10, color=INK, loc='left'); style(ax)
        ax.set_ylabel('D(A, λ) − 197.197  (nat)', fontsize=9, color=INK2)
        ax.set_xlabel('λ / γ_bar  (evaluated points of the golden section; searched λ ∈ [0, 20 γ_bar], shown up to 5 γ_bar)', fontsize=9, color=INK2)
    axs[0].legend(fontsize=8, frameon=False, loc='upper right')
    fig.suptitle('Direct route: D(A, λ) − log det J_C on the discrete model (estimate on the discrete model; points above each frame are clipped; separate y scales)',
                 fontsize=10, color=INK, x=0.01, ha='left')
    fig.savefig(os.path.join(HERE, 'fig_dual_lambda.png'), dpi=130, facecolor=SURF); plt.close(fig)


if __name__ == '__main__':
    {'gamma': fig_gamma, 'dual': fig_dual}[sys.argv[1]]()
