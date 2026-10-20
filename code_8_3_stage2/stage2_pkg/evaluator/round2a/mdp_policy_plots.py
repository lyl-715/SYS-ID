"""Round 2a, B3 figures and policy-shape facts for the 离散 MDP 对偶估算值 (an estimate on the discrete model, not a bound).
Reads results/mdp_policy.npz (policy at the minimising lambda from mdp_estimate.py). Figures:
  fig_mdp_policy.png    optimal action v*(delta)/sigma against delta/sigma at k = 2, 1250, 2500, 4999 (small multiples)
  fig_mdp_occupancy.png distribution of delta_k under that policy at the same k (forward propagation on the same discrete model)
Logs (logs/mdp_policy.log): per k, the delta-intervals where v* = 0, the occupancy mass in 1-2 sigma inside the edges
(18 <= |delta|/sigma <= 19), the mean |delta|/sigma and the most likely |delta|/sigma. Usage: python3 mdp_policy_plots.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import density_recursion_v1 as dr
import mdp_estimate as me

KS = [2, 1250, 2500, 4999]
SURF, INK, INK2, GRID, BLUE, BAND = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#2a78d6', '#d9d8d3'


def zero_intervals(t, v):
    """delta/sigma intervals (consecutive grid nodes) where v* = 0."""
    out = []; i = 0
    while i < len(v):
        if v[i] == 0:
            j = i
            while j+1 < len(v) and v[j+1] == 0: j += 1
            out.append((float(t[i]), float(t[j]))); i = j+1
        else: i += 1
    return out

def style(ax):
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.axvspan(-20, 20, color=BAND, alpha=0.35, lw=0)            # no-send band |delta| < Delta (for an unmoved reference)


if __name__ == '__main__':
    sys.stdout = dr.Tee(os.path.join(HERE, 'logs', 'mdp_policy.log'))
    z = np.load(os.path.join(HERE, 'results', 'mdp_policy.npz')); t = z['t']; pols = z['pols']; l1 = int(z['l1']); lam = float(z['lam'])
    tt, c, q, P, phat, Ninv, N = me.model()
    assert np.array_equal(tt, t)
    print(f"=== mdp_policy_plots.py; policy at lambda* = {lam!r} (离散 MDP 对偶估算值, discrete model only); v_1 = {l1} h")
    # forward occupancy at the selected k
    mu = me.initial_mu(phat, l1); out = np.empty(me.G); used = np.zeros(2*me.L+1, np.bool_); occ = {}
    for k in range(2, dr.K+1):
        if k in KS: occ[k] = mu.copy()
        if k < dr.K:
            pk = pols[k].astype(np.int64); used[:] = False; used[np.unique(pk)+me.L] = True
            me._forward_step(mu, P, q, phat, Ninv, me.M, pk, used, out); mu, out = out, mu
    for k in KS:
        v = pols[k].astype(float)/me.N_SIG; m = occ[k]; ad = np.abs(t)
        zi = zero_intervals(t, pols[k])
        print(f"SHAPE k={k}: v* = 0 on delta/sigma intervals {[(round(a, 3), round(b, 3)) for a, b in zi]}")
        print(f"SHAPE k={k}: occupancy mass with 18 <= |delta|/sigma <= 19 (1-2 sigma inside the edges) {float(m[(ad >= 18) & (ad <= 19)].sum())!r}; "
              f"with |delta|/sigma < 18: {float(m[ad < 18].sum())!r}; with |delta|/sigma > 20: {float(m[ad > 20].sum())!r}")
        print(f"SHAPE k={k}: mean |delta|/sigma {float((m*ad).sum()/m.sum())!r}; most likely delta/sigma {float(t[np.argmax(m)])!r}")
        print(f"SHAPE k={k}: v*/sigma range {float(v.min())!r} to {float(v.max())!r}; send probability under the occupancy {float(m@q)!r}")
    for name, data, ylab in (('fig_mdp_policy.png', {k: pols[k]/me.N_SIG for k in KS}, 'optimal action v* / σ'),
                             ('fig_mdp_occupancy.png', {k: occ[k]*me.N_SIG for k in KS}, 'occupancy density of δ (per σ)')):
        fig, axs = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, sharey=True, facecolor=SURF)
        for ax, k in zip(axs.ravel(), KS):
            style(ax)
            if name == 'fig_mdp_policy.png': ax.step(t, data[k], where='mid', color=BLUE, lw=1.6)
            else: ax.plot(t, data[k], color=BLUE, lw=1.6)
            ax.set_title(f'k = {k}', color=INK, fontsize=10, loc='left')
        for ax in axs[1]: ax.set_xlabel('δ / σ  (δ = r − x_k; shaded: |δ| < Δ = 20σ)', color=INK2, fontsize=9)
        for ax in axs[:, 0]: ax.set_ylabel(ylab, color=INK2, fontsize=9)
        what = 'optimal action' if name == 'fig_mdp_policy.png' else 'occupancy of δ under the policy'
        fig.suptitle(f'Discrete MDP estimate (not a bound): {what} at λ* = {lam:.4f}', color=INK, fontsize=11, x=0.01, ha='left')
        fig.tight_layout(); fig.savefig(os.path.join(HERE, name), dpi=130, facecolor=SURF); plt.close(fig)
        print(f"FIG {name} written")
