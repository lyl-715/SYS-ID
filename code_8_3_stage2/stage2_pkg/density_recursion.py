"""Exact 1-D density recursion for the send-on-delta reference value r_k (reviewer's proposal), numerically on a grid.
f_{k+1}(s) = f_k(s) P_k(s) + g_k(s) [1 - F_k(s+Delta) + F_k(s-Delta)],  g_k = N(x_k, sigma^2) density,
P_k(r) = Phi((r+Delta-x_k)/sigma) - Phi((r-Delta-x_k)/sigma).  Outputs E[N_ev], w_k, J. Compare with MC."""
import numpy as np, time, sys
from scipy.special import ndtr
exec(open('mc83.py').read().split('# --- noise-free')[0])
D = 0.002384694482635785; s = D/20
def run(ngrid_per_sigma=6, margin=8*0.002384694482635785):
    lo, hi = x.min()-margin, x.max()+margin; dz = s/ngrid_per_sigma
    z = np.arange(lo, hi+dz, dz); G = len(z)
    f = np.exp(-0.5*((z-x[0])/s)**2)/(s*np.sqrt(2*np.pi))        # f_2 = g_1
    EN = 1.0; W = np.zeros(K); W[0] = 1.0; mass = []
    sh = int(round(D/dz))                                            # Delta in grid steps (Delta/dz exact? check)
    for k in range(1, K):
        xk = x[k]; al = (z-D-xk)/s; be = (z+D-xk)/s
        P = ndtr(be)-ndtr(al)
        # information weight and expected sends under f (density of r_k)
        ph = lambda u: np.exp(-0.5*u*u)/np.sqrt(2*np.pi)
        cev = 1-P-al*ph(al)+be*ph(be); cno = np.where(P>1e-300, (ph(al)-ph(be))**2/np.maximum(P,1e-300), 0.0)
        W[k] = np.sum(f*(cev+cno))*dz; EN += np.sum(f*(1-P))*dz
        # recursion
        F = np.cumsum(f)*dz                                          # CDF on grid
        Fp = np.concatenate([F[sh:], np.full(sh, F[-1])])             # F(s+Delta)
        Fm = np.concatenate([np.zeros(sh), F[:-sh]])                  # F(s-Delta)
        g = np.exp(-0.5*((z-xk)/s)**2)/(s*np.sqrt(2*np.pi))
        f = f*P + g*(1-Fp+Fm)
        if k % 1000 == 0: mass.append(np.sum(f)*dz)
    J = (H*W[:, None]).T@H/s**2
    return EN, W, J, mass, G, D/dz
for npr in [3, 6, 12]:
    t0 = time.time(); EN, W, J, mass, G, ratio = run(npr)
    print(f"grid {npr}/sigma (G={G}, Delta/dz={ratio:.2f}): E[N]={EN:.4f}  S={W.sum():.3f}  logdet={np.linalg.slogdet(J)[1]:.5f}  mass={np.round(mass,6)}  {time.time()-t0:.1f}s")
print("MC reference (R=10000): E[N]=316.87+-0.02, S=465.96+-0.25, logdet=197.1966+-0.003")
wk = np.load('wk_d20_10000.npy'); EN, W, J, _, _, _ = run(6)
print("max |w_k(recursion) - w_k(MC)| =", np.abs(W-wk).max(), " corr =", np.corrcoef(W, wk)[0,1])
Jmc = (H*wk[:,None]).T@H/s**2
from scipy.linalg import eigh
print("generalized eigenvalues J_rec vs J_mc:", np.round(eigh(J, Jmc, eigvals_only=True), 4))
