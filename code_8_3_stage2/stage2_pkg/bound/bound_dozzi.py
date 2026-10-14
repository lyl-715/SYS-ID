"""Route 1, Script 2: the designed-sampling baseline of Dozzi, Oomen, Gonzalez, "D-Optimized Sampling Design for System
Identification", arXiv:2605.13120 v1 (lit/sources.md #9), for this system, tone set and N = 316.
Run from anywhere:  python3 code_8_3_stage2/stage2_pkg/bound/bound_dozzi.py   (log: bound/bound_dozzi.log)

Relation to Script 1 (review (4)). Dozzi et al. optimise a CONTINUOUS sampling density on [0, T] for i.i.d. random sampling
instants (eq. 15), and only afterwards construct a deterministic sequence by a greedy rule on a grid (Sec. 4.1). The discrete
capped problem (A) of Script 1 (weights on the 1 ms clock grid, each instant at most once) is related to, not identical with,
theirs: their density has no cap (an atom can carry many expected samples), and their sequence is a heuristic, not the
solution of a relaxation.

What the paper specifies, and what is implemented:
  (a) D-optimal sampling density, eq. (15): max_pT log det R0(pT), R0 = N int psi(t) psi(t)^H pT(t) dt, with
      psi(t) = [1, e^{-j w1 t}, e^{j w1 t}, ..., e^{-j w4 t}, e^{j w4 t}] (unit amplitudes: A = diag(a, phases) factors out of det).
      Implemented as a discretisation on the clock grid t_k = k dt (k = 1..5000): a probability vector p, solved with the own
      exchange method (bound_common.exchange, uncapped, W = 1, cutting-plane gap <= 1e-10). psi is replaced by the real
      g(t) = [1, cos w t, sin w t]: psi = U g with a constant invertible U, so the argmax is the same and log det differs by a
      constant. On the grid this is variant (C) of Script 1 divided by 316 (checked against bound/inner_unif.npz). How far the
      continuous optimum can lie above the grid one is bounded by the cutting-plane bound over all t in [0, T]: the gradient
      d(t) = g(t)^T M(p)^-1 g(t) is evaluated on a 0.01 ms grid including t = 0, and U_cont = F + max_t d(t) - 9.
  (b) Deterministic sequence, Sec. 4.1: greedy, t_{k+1} in argmax_{t in grid} s_k(t), s_k(t) = psi(t)^H R_k^{-1} psi(t),
      R_k = sum_{i<=k} psi(t_i) psi(t_i)^H. Implemented exactly with complex psi, R_k^{-1} recomputed each step.
Choices the paper leaves open (fixed here; the alternatives are run as sensitivity):
  - grid {t(1..L)}: the clock grid, L = K = 5000 (1 ms). Its resolution is the minimum inter-sample time.
  - repeated grid points: NOT allowed (without replacement), since the paper says the grid resolution bounds the inter-sample
    time, which holds only if a grid point is used once. Sensitivity W1: with replacement.
  - start: R_k is singular for k < 2M+1 = 9, so s_k is undefined. Main choice G1: the limit eps -> 0 of R_0 = eps I, under
    which argmax s_k(t) = argmax of the residual |(I - P_k) psi(t)|^2 (P_k = projector on span psi(t_1..t_k)), i.e. greedy
    Gram-Schmidt for the first 9 points. Sensitivity G3: R_0 = I (eps = 1).
  - ties (the first pick is a 5000-way tie, |psi(t)|^2 = 9 for all t): values within 1e-12 relative of the max are ties,
    broken by the smallest index (G1: t_1 = 0.001 s). Sensitivity G2: largest index (t_1 = 5.000 s); G4: t_1 forced to 2.5 s.
  - the input: the sequence does not depend on it (A factors out); log det is evaluated with the uniform-input regressors
    H_u and full-sample weights (xi = 1 at the 316 chosen grid points), comparable with J_per(316) and bound (A).
Deterministic: no random numbers. Outputs: bound/bound_dozzi.log, bound/dozzi.json, bound/dozzi.npz, bound/fig_dozzi.png."""
import os, time
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bound_common.py')).read())
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

T0 = time.time()
sys.stdout = Tee(f'{OUT}/bound_dozzi.log')
print(f"Route 1 Script 2 (bound_dozzi.py), {time.strftime('%Y-%m-%d %H:%M:%S')}")
N = 316
INNER = np.load(f'{OUT}/inner_unif.npz'); IJ = json.load(open(f'{OUT}/inner_unif.json'))
LD_A, LD_C = IJ['res']['A']['U'], IJ['res']['C']['U']                 # upper values U (review (3))
print(f"from bound_inner: U_A = {LD_A:.6f}, U_C = {LD_C:.6f}; J_per(316) = {LD_PER:.6f}; J_sod(u_unif) = {LD_SOD:.6f}")

PSI = np.column_stack([np.ones(K)] + [c for m in range(4) for c in (np.exp(-1j*w[m]*t), np.exp(1j*w[m]*t))])   # K x 9
G_unit = np.column_stack([np.ones(K)] + [c for m in range(4) for c in (np.cos(w[m]*t), np.sin(w[m]*t))])      # real, unit amplitude

# --- (a) the D-optimal density ---------------------------------------------------------------------------------------------
print("\n== (a) D-optimal sampling density, eq. (15), on the clock grid ==")
exchange(H_u[::25], 50.0, 1.0, tol=1e-3, max_sweeps=5)    # numba compile
t0 = time.time(); rd = exchange(G_unit, 1.0, np.inf, tol=1e-10, batch=256); td = time.time()-t0
pd = rd['xi']; supp = np.flatnonzero(pd > 0)
_, d_unit = ld_grad(G_unit, pd)
print(f"solved in {td:.2f} s ({rd['sweeps']} sweeps), cutting-plane gap {rd['gap']:.2e} (unit-amplitude real regressors, W = 1)")
print(f"Kiefer-Wolfowitz check on the clock grid: max_k g_k^T M(p)^-1 g_k = {d_unit.max():.10f} (= 9 at the optimum)")
tf = np.linspace(0.0, T, 500001)                          # 0.01 ms, includes t = 0
Gf = np.column_stack([np.ones(tf.size)] + [c for m in range(4) for c in (np.cos(w[m]*tf), np.sin(w[m]*tf))])
Mp = (G_unit*pd[:, None]).T@G_unit; Lp = np.linalg.cholesky(Mp); Vf = np.linalg.solve(Lp, Gf.T); dfine = np.einsum('ij,ij->j', Vf, Vf)
gap_cont = float(dfine.max() - pd@d_unit)
print(f"continuous [0, T] (0.01 ms evaluation grid): max_t d(t) = {dfine.max():.10f} at t = {tf[dfine.argmax()]:.5f} s; "
      f"cutting-plane gap of the grid density for the continuous problem = {gap_cont:.2e} nat")
_in = tf >= dt - 1e-12
print(f"  restricted to [dt, T] (the span of the clock grid, which has no point at t = 0): max_t d(t) = {dfine[_in].max():.10f} "
      f"at t = {tf[_in][dfine[_in].argmax()]:.5f} s, gap {dfine[_in].max() - pd@d_unit:.2e} nat; so the continuous-problem gap comes "
      f"from the endpoint t = 0 that the clock grid t_k = k dt, k = 1..K, does not contain")
print(f"support: {supp.size} grid points; (t s, p): " + '; '.join(f"({t[k]:.3f}, {pd[k]:.4f})" for k in supp))
ld_dens = ld(H_u, N*pd)
print(f"expected FIM of N = 316 i.i.d. draws, E[J] = 316 sigma^-2 sum p_k h_k h_k^T: log det = {ld_dens:.10f}; "
      f"U_C = {LD_C:.10f}; diff {ld_dens-LD_C:+.1e}; max|316 p - xi_C| = {np.abs(N*pd-INNER['C']).max():.2e}")
print(f"continuous density at N = 316: log det <= {ld_dens + gap_cont:.10f} (upper value), i.e. at most {gap_cont:.1e} nat above the grid density")
print(f"largest atom 316 p_max = {N*pd.max():.2f} samples at one grid point: not realisable with distinct full samples")

# --- (b) greedy deterministic sequence ---------------------------------------------------------------------------------------
def greedy(n, eps0=0.0, tie='first', first=None, replace=False, rtol=1e-12):
    """Dozzi Sec. 4.1 greedy on the clock grid; returns chosen grid indices in order."""
    chosen = []; avail = np.ones(K, bool)
    Q = np.zeros((9, 0), complex)
    R = eps0*np.eye(9, dtype=complex)
    for k in range(n):
        if eps0 == 0.0 and k < 9:                          # eps -> 0 limit: residual norm after projection on span psi(t_i)
            proj = PSI.conj() @ Q if Q.shape[1] else np.zeros((K, 0))
            sc = np.real(np.einsum('ij,ij->i', PSI.conj(), PSI)) - np.sum(np.abs(proj)**2, axis=1)
        else:
            Ri = np.linalg.inv(R)
            sc = np.real(np.einsum('ij,ij->i', PSI.conj(), PSI @ Ri.T))
        if not replace: sc = np.where(avail, sc, -np.inf)
        if k == 0 and first is not None: j = first
        else:
            cand = np.flatnonzero(sc >= sc.max() - rtol*abs(sc.max()))
            j = int(cand[0] if tie == 'first' else cand[-1])
        chosen.append(j); avail[j] = False
        v = PSI[j]; R = R + np.outer(v, v.conj())
        if eps0 == 0.0 and k < 9:
            r_ = v - Q @ (Q.conj().T @ v); Q = np.column_stack([Q, r_/np.linalg.norm(r_)])
    return np.array(chosen)

def summarize(idx, label):
    xi = np.bincount(idx, minlength=K).astype(float)
    tt = np.sort(t[idx]); gaps = np.diff(tt)
    ldv = ld(H_u, xi)
    inA = float((INNER['A'][idx] > 0).mean())
    r = dict(label=label, ld=ldv, d_per=ldv-LD_PER, d_A=ldv-LD_A, frac_gap=(ldv-LD_PER)/(LD_A-LD_PER), n_distinct=int(np.unique(idx).size),
             min_gap_ms=float(gaps.min()*1e3), frac_in_A_support=inA, first9=[float(t[j]) for j in idx[:9]],
             w_end=float(((tt <= 0.5) | (tt > 4.5)).mean()))
    print(f"{label:58s} log det {ldv:11.6f}  -J_per {r['d_per']:+.4f}  -A {r['d_A']:+.4f}  share of (A-J_per) {r['frac_gap']:6.3f}  "
          f"distinct {r['n_distinct']:3d}  min gap {r['min_gap_ms']:.0f} ms  in supp(A) {100*inA:5.1f}%  in first/last 0.5 s {100*r['w_end']:.1f}%")
    return r, xi

print("\n== (b) greedy deterministic sequences, N = 316, clock grid ==")
specs = [('G1 main: eps->0 start, ties->smallest index, no replacement', dict()),
         ('G2: eps->0 start, ties->largest index', dict(tie='last')),
         ('G3: ridge start R_0 = I, ties->smallest index', dict(eps0=1.0)),
         ('G4: eps->0 start, t_1 forced to 2.5 s', dict(first=int(np.rint(2.5/dt))-1)),
         ('W1: G1 with replacement (repeats allowed)', dict(replace=True))]
GR = {}; XIG = {}
for lab, kw in specs:
    t0 = time.time(); idx = greedy(N, **kw); el = time.time()-t0
    r, xi = summarize(idx, lab); r['time'] = el; r['kw'] = {k: v for k, v in kw.items()}
    key = lab.split(':')[0].split(' ')[0]; GR[key] = r; XIG[key] = xi; XIG[key+'_order'] = idx
    print(f"    first 9 picks (s): {np.round(r['first9'], 3).tolist()}; time {el:.2f} s")
for key in ('G1',):
    bl = blocks(XIG[key] > 0, t)
    print(f"{key}: {len(bl)} blocks of consecutive grid points; lengths min/median/max = {min(b[2] for b in bl)}/"
          f"{int(np.median([b[2] for b in bl]))}/{max(b[2] for b in bl)}")
    m_ = XIG[key] > 0; mA = INNER['A'] > 0
    print(f"{key} vs supp(A): |G1| = {int(m_.sum())}, |supp A| = {int(mA.sum())}, overlap {int((m_ & mA).sum())}; "
          f"sum_k min(xi_G1, xi_A) = {np.minimum(XIG[key], INNER['A']).sum():.2f} of 316")
print(f"spread of log det over G1-G4 (without replacement): {max(GR[k]['ld'] for k in ('G1','G2','G3','G4'))-min(GR[k]['ld'] for k in ('G1','G2','G3','G4')):.4f} nat")

print("\n== Table 2: Dozzi design between J_per(316) and U_A (log det, nat; bounds quoted as upper values U; ratio = e^{-diff/18}) ==")
rows = [('J_per(316)', LD_PER), ('Dozzi greedy G1 (main)', GR['G1']['ld'])] + \
       [(f'Dozzi greedy {k} (sensitivity)', GR[k]['ld']) for k in ('G2', 'G3', 'G4', 'W1')] + \
       [('U_A (W=316, cap 1)', LD_A), ('Dozzi density E[J], N=316, grid (= U_C)', ld_dens),
        ('Dozzi density E[J], N=316, continuous (U)', ld_dens + gap_cont), ('J_sod(u_unif)', LD_SOD)]
print(f"{'row':42s} {'log det':>12s} {'- J_per':>9s} {'- A':>9s} {'- J_sod':>9s} {'ratio vs J_sod':>15s}")
for nm, v in rows:
    print(f"{nm:42s} {v:12.6f} {v-LD_PER:+9.4f} {v-LD_A:+9.4f} {v-LD_SOD:+9.4f} {axis_ratio(v-LD_SOD):15.4f}")

# --- figure ---------------------------------------------------------------------------------------------------------------
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(3, 1, figsize=(10, 6.6), sharex=True)
ax[0].fill_between(t, 0, INNER['A'], step='mid', color=C2, lw=0, label=r'relaxation (A): optimal $\xi_k$')
tg = t[XIG['G1'] > 0]
ax[0].vlines(tg, 1.12, 1.32, color=C1, lw=0.5, label='Dozzi greedy G1: sample times')
ax[0].set_ylim(0, 1.36); ax[0].set_yticks([0, 0.5, 1]); ax[0].set_ylabel(r'$\xi_k$')
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False, fontsize=8)
ax[0].set_title('Designed sampling at N = 316: relaxation (A), the Dozzi greedy sequence and the D-optimal density (clock grid)', color=INK, loc='left', fontsize=10)
ax[1].vlines(t[supp], 0, pd[supp], color=C3, lw=1.6); ax[1].plot(t[supp], pd[supp], 'o', color=C3, ms=3)
ax[1].set_ylabel('density atoms $p_k$'); ax[1].set_title(f'D-optimal density, eq. (15), on the clock grid: {supp.size} atoms', loc='right', color=INK2, fontsize=8.5)
edges = np.arange(0, T+1e-9, 0.1)
for lab_, xi_, col, ls in [('periodic (316)', None, INK2, '--'), ('Dozzi greedy G1', XIG['G1'], C1, '-'), ('relaxation (A)', INNER['A'], C2, '-')]:
    if xi_ is None: cnt = np.histogram(ti, edges)[0].astype(float)
    else: cnt = np.histogram(t, edges, weights=xi_)[0]
    ax[2].step(edges, np.append(cnt, cnt[-1]), where='post', color=col, lw=1.2, ls=ls, label=lab_)
ax[2].set_ylabel('weight per 100 ms'); ax[2].set_xlabel('t (s)'); ax[2].legend(loc='upper center', ncol=3, frameon=False, fontsize=8)
for a_ in ax: a_.grid(True, color=GRID, lw=0.6); a_.set_xlim(0, T)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_dozzi.png', dpi=150); plt.close(fig)
print(f"\nfigure: {OUT}/fig_dozzi.png")

json.dump(dict(density=dict(support=[(float(t[k]), float(pd[k])) for k in supp], gap=rd['gap'], ld_N316=ld_dens, time=td,
                              gap_cont=gap_cont, U_cont_N316=ld_dens+gap_cont),
               greedy=GR), open(f'{OUT}/dozzi.json', 'w'), indent=1, default=float)
np.savez_compressed(f'{OUT}/dozzi.npz', p_density=pd, **XIG)
log_runtime('bound_dozzi', T0, 'Dozzi density (eq. 15) and greedy sequence (Sec. 4.1), N = 316, 4 start variants + with replacement')
