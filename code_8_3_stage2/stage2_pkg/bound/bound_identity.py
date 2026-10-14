"""Route 1, Script 3 (replaces the outer search, review (1) and (2)): the input-factorisation identity of the relaxation, and
the free-placement relaxation gap of the seven stage-2 finalists at their own total weight S(u).
Run from anywhere:  python3 code_8_3_stage2/stage2_pkg/bound/bound_identity.py   (log: bound/bound_identity.log)

Statement. For every input u = (a, chi), a_m = sqrt(p_m), the regressor matrix of regressors(a, chi, t) factorises as
    H(u) = H(u_unif) T(u),   T(u) = blockdiag(1, (a_1/0.5) R(chi_1), ..., (a_4/0.5) R(chi_4)),  R(c) = [[cos c, -sin c], [sin c, cos c]],
so det T(u) = prod_m 4 p_m and, for every fixed weight vector xi,
    F_u(xi) := log det(sigma^-2 H(u)^T diag(xi) H(u)) = F_unif(xi) + 2 sum_m log(4 p_m).
The shift does not depend on xi, so the maximiser over any fixed feasible set Xi_W is the same for every input, and
    U_W(u) = U_W(u_unif) + 2 sum_m log(4 p_m)   for every fixed W (with or without the cap):
phase-independent, and maximised exactly at uniform power (Jensen: sum_m log(4 p_m) <= 4 log(sum_m p_m) = 0, equality iff
p_m = 1/4). An outer search over the input is therefore unnecessary.

Checks:
  (1) the factorisation itself: max |H(u) - H(u_unif) T(u)| and det T(u) vs prod 4 p_m;
  (2) the log-det identity at random FIXED weights: residual F_u(xi) - F_unif(xi) - 2 sum log(4 p_m), for random non-uniform
      p (Dirichlet(1,1,1,1)), random non-zero phases (uniform on [0, 2 pi)) and random feasible xi (capped with W = 316 and
      W = S(u_unif); uncapped with W = 316), plus the seven finalists' inputs; RNG seed 20261022 (the only random numbers of
      Route 1);
  (3) the identity at the optimum: U_316(u) solved directly (cold start, cutting-plane gap <= 1e-9) at 4 random inputs and at
      the 7 finalists, against U_316(u_unif) + 2 sum log(4 p_m).
Finalists (review (2)): for each, S(u) = sum_k w_k from the batch-1 weights (stage2/final_qxw.npz; R = 4000, seed 20261013),
ld J_sod(u) from the same weights (checked against stage2/final_J.npz), and the relaxation solved with W = S(u), cap 1, at
input u (cold start). Reported: F, the cutting-plane upper value U_{S(u)}(u) and its gap, and the
"free-placement relaxation gap at equal total weight"  ld J_sod(u) - U_{S(u)}(u)  (<= 0, since J_sod's weights are feasible).
Outputs: bound/bound_identity.log, bound/identity.json."""
import os, time
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bound_common.py')).read())

T0 = time.time()
sys.stdout = Tee(f'{OUT}/bound_identity.log')
print(f"Route 1 Script 3 (bound_identity.py), {time.strftime('%Y-%m-%d %H:%M:%S')}")
IJ = json.load(open(f'{OUT}/inner_unif.json')); INNER = np.load(f'{OUT}/inner_unif.npz')
U_A_unif = IJ['res']['A']['U']
print(f"from bound_inner: U_316(u_unif) = {U_A_unif:.10f} (gap {IJ['res']['A']['gap']:.1e}), U_S(u_unif) = {IJ['res']['B']['U']:.10f} "
      f"(W = S(u_unif) = {W_B:.6f})")

def Tmat(a, chi):
    T_ = np.zeros((P, P)); T_[0, 0] = 1.0
    for m in range(4):
        c, s_ = np.cos(chi[m]), np.sin(chi[m]); T_[1+2*m:3+2*m, 1+2*m:3+2*m] = (a[m]/0.5)*np.array([[c, -s_], [s_, c]])
    return T_
def shift(p): return float(2*np.sum(np.log(4*np.asarray(p))))

F1 = json.load(open('stage2/final_batch1.json')); QXW = np.load('stage2/final_qxw.npz'); FJ = np.load('stage2/final_J.npz')
FIN = [(r['id'], np.asarray(r['a']), np.asarray(r['chi']), r) for r in F1['records']]

# --- (1) factorisation ----------------------------------------------------------------------------------------------------
rng = np.random.default_rng(20261022)
NR = 20
Prand = rng.dirichlet(np.ones(4), NR); CHIrand = rng.uniform(0, TWO_PI, (NR, 4))
inputs = [(f'R{i:02d}', np.sqrt(Prand[i]), CHIrand[i]) for i in range(NR)] + [(fid, a_, c_) for fid, a_, c_, _ in FIN]
print(f"\n== (1) factorisation H(u) = H(u_unif) T(u): {NR} random inputs (seed 20261022: p ~ Dirichlet(1,1,1,1), chi ~ U[0, 2pi)^4) "
      f"+ 7 finalists ==")
r1 = []
for nm, a_, c_ in inputs:
    Hu = regressors(a_, c_, t); Tm = Tmat(a_, c_)
    r1.append((np.abs(Hu - H_u@Tm).max(), abs(np.linalg.det(Tm)/np.prod(4*a_**2) - 1)))
print(f"max_u max_k,j |H(u) - H(u_unif) T(u)| = {max(v[0] for v in r1):.2e};  max_u |det T(u) / prod(4 p_m) - 1| = {max(v[1] for v in r1):.2e}")

# --- (2) log-det identity at random fixed weights --------------------------------------------------------------------------
print("\n== (2) F_u(xi) - F_unif(xi) - 2 sum log(4 p_m) at random fixed feasible xi (same RNG stream) ==")
rows2 = []
for nm, a_, c_ in inputs:
    Hu = regressors(a_, c_, t); sh = shift(a_**2)
    for lab, Wv, cap in [('capped W=316', 316.0, 1.0), ('capped W=S(u_unif)', W_B, 1.0), ('uncapped W=316', 316.0, np.inf)]:
        y = rng.uniform(0, 1, K) if np.isfinite(cap) else rng.exponential(1.0, K)
        xi = proj_capped(y*Wv/y.sum(), Wv, cap) if np.isfinite(cap) else y*Wv/y.sum()
        res_ = ld(Hu, xi) - ld(H_u, xi) - sh
        rows2.append(dict(u=nm, xi=lab, shift=sh, resid=res_, F_u=ld(Hu, xi)))
for nm, a_, c_ in inputs[:5] + inputs[NR:]:
    rr = [r for r in rows2 if r['u'] == nm]
    print(f"{nm:14s} p = ({','.join(f'{v:.3f}' for v in a_**2)}) chi = ({','.join(f'{v:.2f}' for v in np.mod(c_, TWO_PI))}) "
          f"2 sum log 4p = {rr[0]['shift']:+.6f}; residuals: " + ', '.join(f"{r['xi']}: {r['resid']:+.1e}" for r in rr))
print(f"... ({NR-5} more random inputs in identity.json)")
print(f"max |residual| over {len(rows2)} (input, xi) pairs = {max(abs(r['resid']) for r in rows2):.2e} nat "
      f"(log det values ~ 190: double-precision level)")

# --- (3) identity at the optimum (cold solves) -----------------------------------------------------------------------------
exchange(H_u[::25], 50.0, 1.0, tol=1e-3, max_sweeps=5)    # numba compile
print("\n== (3) U_316(u) solved directly at input u (cold start, cutting-plane gap <= 1e-9) vs U_316(u_unif) + 2 sum log(4 p_m) ==")
rows3 = []
for nm, a_, c_ in inputs[:4] + inputs[NR:]:
    Hu = regressors(a_, c_, t); t0 = time.time(); rx = exchange(Hu, 316.0, 1.0, tol=1e-9, batch=256); el = time.time()-t0
    pred = U_A_unif + shift(a_**2)
    cross = ld(H_u, rx['xi']) - IJ['res']['A']['F']           # xi*(u) evaluated at the uniform input
    rows3.append(dict(u=nm, U=rx['ub'], F=rx['f'], gap=rx['gap'], pred=pred, diff=rx['ub']-pred, time=el,
                      dxi=float(np.abs(rx['xi']-INNER['A']).max()), cross=cross))
    print(f"{nm:14s} U_316(u) = {rx['ub']:.9f} (F {rx['f']:.9f}, gap {rx['gap']:.1e}, {el:5.1f} s); predicted {pred:.9f}; "
          f"diff {rx['ub']-pred:+.1e}; max|xi*(u) - xi*(u_unif)| = {rows3[-1]['dxi']:.1e}; F_unif(xi*(u)) - F_unif(xi*(u_unif)) = {cross:+.1e}")
print(f"max |U_316(u) - prediction| = {max(abs(r['diff']) for r in rows3):.1e} nat (each side carries a cutting-plane gap <= 1e-9)")
print("Phases: the 7 finalists and 4 random inputs have non-zero phases; the prediction uses only p, and matches.")
print(f"Maximisers: xi*(u) from a cold solve at u differs from xi*(u_unif) by up to {max(r['dxi'] for r in rows3):.1e} per weight, but "
      f"evaluated at the uniform input it is within {max(abs(r['cross']) for r in rows3):.1e} nat of the optimum: the objective is flat"
      " near the optimum, and both are maximisers to solver tolerance.")

# --- finalists: free-placement relaxation gap at equal total weight --------------------------------------------------------
print("\n== (4) stage-2 finalists: S(u) from batch-1 weights (R = 4000, seed 20261013); relaxation at W = S(u), cap 1, input u ==")
rows4 = []
for fid, a_, c_, rec in FIN:
    Hu = regressors(a_, c_, t); wk = QXW[f'{fid}_w']; S = float(wk.sum())
    Jsod = (Hu*wk[:, None]).T@Hu/s**2; ldJ = float(np.linalg.slogdet(Jsod)[1])
    ldJ_store = float(np.linalg.slogdet(FJ[fid])[1]); ld_rec = F1['ldC'] + rec['dmD_C']
    se_ld = float(np.std(rec['ldb'], ddof=1)/np.sqrt(len(rec['ldb'])))
    t0 = time.time(); rx = exchange(Hu, S, 1.0, tol=1e-10, batch=256); el = time.time()-t0
    kk = kkt_summary(rx['xi'], ld_grad(Hu, rx['xi'])[1], 1.0)
    rows4.append(dict(id=fid, p=(a_**2).tolist(), chi=np.mod(c_, TWO_PI).tolist(), S=S, wmax=float(wk.max()), ld_sod=ldJ, ld_sod_se=se_ld,
                      ld_sod_stored=ldJ_store, ld_sod_record=ld_rec, F=rx['f'], U=rx['ub'], gap=rx['gap'], time=el, nu=kk['nu'],
                      n_interior=kk['n_interior'], relax_gap=ldJ-rx['ub'], shift=shift(a_**2), dmD_C=rec['dmD_C']))
    print(f"{fid:14s} S(u) = {S:9.4f} (max w {wk.max():.4f}); ld J_sod(u) = {ldJ:.6f} (+- {se_ld:.4f} batch se; stored final_J "
          f"{ldJ_store-ldJ:+.1e}; ldC+dmD_C {ld_rec-ldJ:+.1e}); U_S(u)(u) = {rx['ub']:.6f} (F {rx['f']:.6f}, gap {rx['gap']:.1e}, "
          f"{el:.1f} s, {kk['n_interior']} interior, nu {kk['nu']:.5f})")
print("\nTable 3. Free-placement relaxation gap at equal total weight, ld J_sod(u) - U_{S(u)}(u) (nat), and the axis ratio e^{-gap/18}")
print(f"{'id':14s} {'Delta m_D^(C)':>13s} {'S(u)':>9s} {'2sum log4p':>11s} {'ld J_sod(u)':>12s} {'U_S(u)(u)':>12s} {'gap':>9s} {'ratio':>7s}")
for r in rows4:
    print(f"{r['id']:14s} {r['dmD_C']:+13.4f} {r['S']:9.3f} {r['shift']:+11.5f} {r['ld_sod']:12.4f} {r['U']:12.4f} {r['relax_gap']:+9.4f} "
          f"{axis_ratio(r['relax_gap']):7.4f}")
ru = next(r for r in rows4 if r['id'] == 'UNIF')
print("\nTable 4. Decomposition against UNIF (same batch-1 draws): Delta ld J_sod = Delta U_{S(u)} + Delta gap, and Delta U = (shift 2 sum log 4p) + (effect of S(u))")
print(f"{'id':14s} {'Delta S':>9s} {'Delta ldJ':>10s} {'Delta U':>9s} {'of which p':>11s} {'of which S':>11s} {'Delta gap':>10s}")
for r in rows4:
    dU = r['U']-ru['U']
    print(f"{r['id']:14s} {r['S']-ru['S']:+9.3f} {r['ld_sod']-ru['ld_sod']:+10.4f} {dU:+9.4f} {r['shift']:+11.5f} {dU-r['shift']:+11.4f} "
          f"{r['relax_gap']-ru['relax_gap']:+10.4f}")
print(f"(nu at the finalists: {min(r['nu'] for r in rows4):.5f} to {max(r['nu'] for r in rows4):.5f} nat per unit of W, so an error of 0.1 in"
      f" the Monte Carlo estimate S(u) moves U_S(u)(u) by about {0.1*max(r['nu'] for r in rows4):.4f} nat)")
print(f"UNIF at R = 10000 (bound_inner, S = {W_B:.3f}): ld J_sod - U_S = {LD_SOD-IJ['res']['B']['U']:+.4f}; at batch 1 (S = {ru['S']:.3f}): {ru['relax_gap']:+.4f}")
print("\nU_{S(u)}(u) = U_{S(u)}(u_unif) + 2 sum log(4 p_m) depends on the input only through p (the shift) and through S(u)"
      " (U_W(u_unif) increases with W, at the rate nu per unit of weight). The value at S(u_unif) is therefore not a bound for"
      " other inputs; each finalist is compared with the relaxation at its own S(u).")

json.dump(dict(seed=20261022, n_random=NR, factorisation=dict(max_H_resid=max(v[0] for v in r1), max_detT_rel=max(v[1] for v in r1)),
               fixed_xi=rows2, optimum=rows3, finalists=rows4), open(f'{OUT}/identity.json', 'w'), indent=1, default=float)
log_runtime('bound_identity', T0, f'identity at {len(rows2)} random fixed xi (seed 20261022), {len(rows3)} cold optimum checks, 7 finalists at W = S(u)')
