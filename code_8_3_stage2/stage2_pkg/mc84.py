"""§8.4: Monte Carlo check of the CRLB for send-on-delta data (uniform input, Delta_*, sigma=Delta_*/20).
Estimator: maximum likelihood of the censored-Gaussian model of §5 (events = full Gaussian observation,
non-events = Pr{|z_k - r_k| < Delta}), theta in R^9. Also the naive LS on sent samples only.
Compare sample covariance over R runs with J^{-1} (J from the 10000-run weights of mc837.py)."""
import numpy as np, time, json, sys, warnings
warnings.filterwarnings("ignore")   # the unused tail branch of np.where may evaluate log1p(-1); the selected branch is finite
from scipy.special import ndtr, log_ndtr
from scipy.optimize import minimize
exec(open('mc83.py').read().split('# --- noise-free')[0])   # x, H, K, T, theta0 pieces, sod(), cfun()
D = 0.002384694482635785; s = D/20
theta0 = np.concatenate([[G(0)*a0], np.ravel(np.column_stack([Gj.real, Gj.imag]))])   # [G0, Re1, Im1, ..., Re4, Im4]
assert np.allclose(H@theta0, x)
R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
seed = 20261009                                   # independent of the seeds used for J (20261006) and B
rng = np.random.default_rng(seed); Z = rng.standard_normal((R, K))

def phi(u): return np.exp(-0.5*u*u)/np.sqrt(2*np.pi)
def nll_grad_hess(theta, z, ev, r):
    """Negative log-likelihood of the send-on-delta data, its gradient and (observed) Hessian in theta.
    ev: boolean events (k=1 included); r: pre-update reference (r[0] unused)."""
    mu = H@theta; ne = ~ev
    res = (z[ev]-mu[ev])/s
    nll = 0.5*np.sum(res**2)
    g_mu = np.zeros(K); h_mu = np.zeros(K)
    g_mu[ev] = -res/s; h_mu[ev] = 1.0/s**2
    al = (r[ne]-D-mu[ne])/s; be = (r[ne]+D-mu[ne])/s
    lo = log_ndtr(be) + np.log1p(-np.exp(log_ndtr(al)-log_ndtr(be)))      # log P via lower tail
    up = log_ndtr(-al) + np.log1p(-np.exp(log_ndtr(-be)-log_ndtr(-al)))   # log P via upper tail
    logP = np.where(al+be < 0, lo, up)
    nll -= np.sum(logP)
    ea = np.exp(-0.5*al*al-0.5*np.log(2*np.pi)-logP)                     # phi(al)/P
    eb = np.exp(-0.5*be*be-0.5*np.log(2*np.pi)-logP)                     # phi(be)/P
    g_mu[ne] = -(ea-eb)/s                                                 # -(d logP/d mu)
    h_mu[ne] = ((be*eb-al*ea) + (ea-eb)**2)/s**2                          # -(d2 logP/d mu2) >= 0 (log-concave)
    return nll, H.T@g_mu, (H*h_mu[:, None]).T@H

def mle(z, ev, r, maxit=50):
    th_ls = np.linalg.lstsq(H[ev], z[ev], rcond=None)[0]        # LS on sent samples: initial value and naive estimator
    th = th_ls.copy(); f, g, Hs = nll_grad_hess(th, z, ev, r); it = 0; conv = False
    for it in range(1, maxit+1):
        step = np.linalg.solve(Hs, g); t = 1.0
        while True:                                              # backtracking line search on the concave objective
            f_new, g_new, Hs_new = nll_grad_hess(th-t*step, z, ev, r)
            if f_new <= f - 1e-4*t*(g@step) or t < 1e-8: break
            t *= 0.5
        th = th-t*step; dec = g@step
        f, g, Hs = f_new, g_new, Hs_new
        if dec < 1e-9*max(1.0, abs(f)): conv = True; break         # Newton decrement small -> converged
    return th, th_ls, conv, it

t0 = time.time(); TH = np.zeros((R, 9)); TL = np.zeros((R, 9)); ok = 0; nits = []
for i in range(R):
    z = x + s*Z[i]; eps, r, _ = sod(z, D)
    TH[i], TL[i], suc, nit = mle(z, eps, r); ok += suc; nits.append(nit)
print(f"R={R}, converged {ok}/{R}, mean iters {np.mean(nits):.1f}, time {time.time()-t0:.0f}s")

# reference: J from the 10000-run weights (mc837.py), CRLB = J^{-1}
wk = np.load('wk_d20_10000.npy'); J = (H*wk[:, None]).T@H/s**2; C = np.linalg.inv(J)
names = ['G0', 'Re1', 'Im1', 'Re2', 'Im2', 'Re3', 'Im3', 'Re4', 'Im4']
def report(TH, label):
    bias = TH.mean(0)-theta0; S = np.cov(TH.T, ddof=1); sd = np.sqrt(np.diag(S)); crlb = np.sqrt(np.diag(C))
    ratio = sd/crlb; tstat = bias/(sd/np.sqrt(R))
    # batch se of the sd ratio (10 batches)
    B = 10; rb = np.array([np.sqrt(np.var(TH[b*R//B:(b+1)*R//B], axis=0, ddof=1))/crlb for b in range(B)])
    ratio_se = rb.std(0, ddof=1)/np.sqrt(B)
    print(f"\n[{label}]")
    print(" param   sd(MC)      sqrt(CRLB)  ratio   (batch se)   bias/sd  t")
    for n, a_, b_, c_, d_, e_ in zip(names, sd, crlb, ratio, ratio_se, bias/sd):
        print(f" {n:4s}  {a_:.3e}  {b_:.3e}  {c_:.3f}  ({d_:.3f})     {e_:+.3f}  {e_*np.sqrt(R):+.1f}")
    ld = np.linalg.slogdet(S)[1]-np.linalg.slogdet(C)[1]
    # whitened: eigenvalues of C^{-1/2} S C^{-1/2} = generalized eig of S vs C ; all ~1 if CRLB attained
    lam = np.linalg.eigvalsh(np.linalg.solve(C, S)) if False else np.sort(np.real(np.linalg.eigvals(np.linalg.solve(C, S))))
    print(f" logdet(Sigma_MC) - logdet(J^-1) = {ld:+.3f}  (per-axis geometric mean ratio {np.exp(ld/18):.3f})")
    print(f" eigenvalues of J^(1/2) Sigma_MC J^(1/2): {np.round(lam,3)}")
    return dict(sd=sd.tolist(), crlb=crlb.tolist(), ratio=ratio.tolist(), ratio_se=ratio_se.tolist(), bias=bias.tolist(), t=(bias/(sd/np.sqrt(R))).tolist(), logdet_diff=float(ld), lam=lam.tolist())
out = dict(R=R, seed=seed, mle=report(TH, 'MLE, censored likelihood'), ls_sent=report(TL, 'LS on sent samples only'))
json.dump(out, open('mc84.json', 'w'), indent=1); np.save('theta_mle.npy', TH); np.save('theta_ls.npy', TL)

# --- finite-sample benchmark for the whitened eigenvalue spread (Gaussian approximation) ---
# If theta_hat ~ N(theta0, J^{-1}), then W = J^{1/2} Sigma_hat J^{1/2} ~ Wishart_9(R-1, I)/(R-1).
# Tail probability of the observed max_i |lambda_i(W)-1| under this benchmark; 5000 draws, seed 1.
Sg = np.cov(TH.T, ddof=1); L = np.linalg.cholesky(J); W = L.T@Sg@L
obs = np.max(np.abs(np.linalg.eigvalsh(W)-1))
rngW = np.random.default_rng(1); M = 5000; p = 9; tail = 0
for _ in range(M):
    X = rngW.standard_normal((R-1, p)); lam = np.linalg.eigvalsh(X.T@X/(R-1)); tail += np.max(np.abs(lam-1)) >= obs
print(f"whitened eigenvalues: max|lambda-1| = {obs:.4f}; Wishart benchmark tail probability = {tail/M:.3f} ({M} draws, seed 1)")
# Hotelling T^2 for joint bias
m = TH.mean(0)-theta0; T2 = R*m@np.linalg.solve(Sg, m)
from scipy.stats import f as fdist
print(f"Hotelling T2 = {T2:.3f}, p = {1-fdist.cdf((R-p)/(p*(R-1))*T2, p, R-p):.3f}")
