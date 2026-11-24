#!/usr/bin/env python3
"""Numerical checks for Section IV-A (upper limit over interval triggers).
Independent of the repository; model and c() copied from verify_lebesgue/verify.py.
 (1) Lemma: at fixed send probability the symmetric window maximises c; derivative formula.
 (2) c_max(q) = q + 2 a phi(a), d c_max / dq = a^2, concavity.
 (3) Convex allocation problem: optimum, bracketing, optimality condition a_k^2 gamma_k = lambda.
 (4) Constant half-width (scheme of You et al.) at the same budget.
"""
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.optimize import brentq

om = np.array([3*np.sqrt(2), 3*np.pi, 12.0, 12.3])
a0, am = 1.0, 0.5
dt, K = 1e-3, 5000
G = lambda s: 2.0/(1.0+s)**2
theta = np.concatenate([[G(0).real], np.ravel([[G(1j*w).real, G(1j*w).imag] for w in om])])
t = dt*np.arange(1, K+1)
H = np.ones((K, 9)); H[:, 0] = a0
for i, w in enumerate(om):
    H[:, 1+2*i] = am*np.cos(w*t); H[:, 2+2*i] = -am*np.sin(w*t)
DELTA = 0.002384694; sig = DELTA/20.0; p = 9
B = 316.8922            # E[N] of constant SoD from verify_lebesgue (sigma/24 grid)
phi = lambda v: np.exp(-0.5*v*v)/np.sqrt(2*np.pi)

def c_ab(al, be):
    P = ndtr(be) - ndtr(al)
    afa = al*phi(al) if np.isfinite(al) else 0.0
    bfb = be*phi(be) if np.isfinite(be) else 0.0
    return 1 - P - afa + bfb + (phi(al)-phi(be))**2/P

print("== (1) symmetric window is best at fixed send probability ==")
for q in [0.02, 0.063, 0.2, 0.5]:
    P = 1 - q
    a = -ndtri(q/2)
    al_max = ndtri(q)                     # one-sided limit (beta = inf)
    als = np.linspace(-6, al_max - 1e-9, 20001)
    bes = ndtri(np.minimum(ndtr(als) + P, 1 - 1e-16))
    cs = np.array([c_ab(x, y) for x, y in zip(als, bes)])
    i = np.argmax(cs)
    # derivative formula: dc/dalpha = phi(alpha) (beta-alpha) (2 mu - alpha - beta)
    j = 6000; h = 1e-5
    be = lambda al: ndtri(ndtr(al) + P)
    num = (c_ab(als[j]+h, be(als[j]+h)) - c_ab(als[j]-h, be(als[j]-h)))/(2*h)
    mu = (phi(als[j]) - phi(bes[j]))/P
    ana = phi(als[j])*(bes[j]-als[j])*(2*mu - als[j] - bes[j])
    print(f" q={q:5.3f}: argmax alpha={als[i]:+.4f} (-a={-a:+.4f}), max c={cs[i]:.6f}, "
          f"c_max formula={q+2*a*phi(a):.6f}, one-sided c={c_ab(al_max, np.inf):.4f}; "
          f"dc/dalpha num={num:+.3e} formula={ana:+.3e}")
print(" table at q=6.3%:", [round(c_ab(*w), 3) for w in [(-1.86, 1.86), (-1.60, 2.39), (-1.53, np.inf)]])

print("== (2) slope of c_max ==")
cmax = lambda q: q + 2*(-ndtri(q/2))*phi(-ndtri(q/2))
for q in [0.02, 0.063, 0.2, 0.5]:
    h = 1e-6; a = -ndtri(q/2)
    print(f" q={q}: a={a:.3f} c_max={cmax(q):.4f} slope num={(cmax(q+h)-cmax(q-h))/(2*h):.4f} a^2={a*a:.4f}")

print("== (3) allocation problem ==")
def Jbar(a):                               # a_k for k>=2 (index 1..K-1); step 1 always sent
    w = np.ones(K); w[1:] = 2*ndtr(-a) + 2*a*phi(a)
    return (H*w[:, None]).T @ H / sig**2
def ld(a): return np.linalg.slogdet(Jbar(a))[1]
def budget(a): return 1 + np.sum(2*ndtr(-a))

# fixed point on the optimality condition a_k^2 gamma_k = lambda
a = np.full(K-1, 1.86)
for it in range(200):
    A = np.linalg.inv(Jbar(a))
    gam = np.einsum('ki,ij,kj->k', H[1:], A, H[1:])/sig**2
    f = lambda lam: 1 + np.sum(2*ndtr(-np.sqrt(lam/gam))) - B
    lam = brentq(f, 1e-8, 1e3)
    a_new = np.sqrt(lam/gam)
    step = np.max(np.abs(a_new - a))
    a = 0.5*a + 0.5*a_new
    if step < 1e-11: break
A = np.linalg.inv(Jbar(a)); gam = np.einsum('ki,ij,kj->k', H[1:], A, H[1:])/sig**2
val = ld(a)
print(f" iterations {it}, E[N]={budget(a):.6f} (B={B}), log det = {val:.5f}")
print(f" a_k range {a.min():.3f}..{a.max():.3f}; q_k range {100*2*ndtr(-a.max()):.2f}%..{100*2*ndtr(-a.min()):.2f}%")
print(f" optimality condition: a_k^2 gamma_k min/max = {np.min(a*a*gam):.6e} / {np.max(a*a*gam):.6e}")
# upper bracket: concave objective <= tangent plane at q_hat, maximised over {sum q <= B-1, 0<=q<=1}
qhat = 2*ndtr(-a); grad = a*a*gam           # d logdet / d q_k = gamma_k * dc_max/dq = gamma_k a_k^2
order = np.argsort(-grad); rem = B - 1; lin = 0.0
for k in order:
    take = min(1.0, rem); lin += grad[k]*(take - qhat[k]); rem -= take
    if rem <= 0:
        lin += -np.sum(grad[order[np.where(order == k)[0][0]+1:]]*qhat[order[np.where(order == k)[0][0]+1:]]); break
print(f" bracket: [{val:.5f}, {val+lin:.5f}]")

print("== (4) constant half-width at the same budget ==")
ac = brentq(lambda v: budget(np.full(K-1, v)) - B, 0.5, 4)
print(f" a={ac:.4f}, log det = {ld(np.full(K-1, ac)):.5f}, below optimum by {val - ld(np.full(K-1, ac)):.4f}")
print(f" all samples sent: log det = {np.linalg.slogdet(H.T@H/sig**2)[1]:.4f}")
print(f" optimum minus constant SoD (197.2010): {val-197.2010:.4f}")
np.save('a_opt.npy', a)
