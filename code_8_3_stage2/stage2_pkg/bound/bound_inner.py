"""Route 1, Script 1: the free-weight D-optimal relaxation for a fixed input (uniform input a_m = 1/2, chi = 0).
Run from anywhere:  python3 code_8_3_stage2/stage2_pkg/bound/bound_inner.py   (log: bound/bound_inner.log)

    max_xi  log det( sigma^-2 sum_k xi_k h_k h_k^T )   s.t. 0 <= xi_k <= cap,  sum_k xi_k = W,   k = 1..K = 5000

  (A) W = 316, cap 1      the relaxation bound for 316 full measurements on the clock grid, each instant at most once
  (B) W = S(u_unif), cap 1  the free-placement relaxation at the total information weight of send-on-delta at this input,
                          S(u_unif) = sum_k w_k = 465.958 (wk_d20_10000.npy). Review (2): B uses the input's own S(u), not 466.
  (C) W = 316, no cap     Kiefer-Wolfowitz design (support <= 45 points), reference

Solvers: (i) own batched pairwise-exchange method (`exchange`, bound_common.py), cold start from uniform weights, stopped at
cutting-plane gap <= 1e-10 nat; (ii) own spectral projected gradient (`spg`), cold start, stopped at gap <= 1e-8 or 20000 its;
(iii) cvxpy log_det with Clarabel (default settings) and SCS (eps 1e-9). Every solution is projected onto the feasible set
and certified (review (3)): F(xi~), the cutting-plane upper bound U = F(xi~) + max_{xi in Xi_W} grad F(xi~)^T (xi - xi~)
(an LP over the box-and-sum set, solved by sorting) and its gap; the duality bound is printed as a secondary check.
Only the upper value U is quoted as the bound. Deterministic: no random numbers are drawn.
Outputs: bound/bound_inner.log, bound/inner_unif.json, bound/inner_unif.npz, bound/fig_inner_A.png."""
import os, time
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bound_common.py')).read())
from scipy.optimize import brentq
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

T0 = time.time()
sys.stdout = Tee(f'{OUT}/bound_inner.log')
print(f"Route 1 Script 1 (bound_inner.py), {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"setup: K={K}, dt={dt}, T={T}, Delta_*={D!r}, sigma=Delta_*/20={s!r}, tones w={np.round(w, 6).tolist()}, a0={a0}")
print(f"uniform input a={a_u.tolist()}, chi={chi_u.tolist()}; regressors(a, chi, t) of mc85_scan.py; max|H_u - mc83.H| = {np.abs(H_u-H).max():.1e}")

# --- references -----------------------------------------------------------------------------------------------------------
print("\n== references ==")
print(f"log det J_per(316)   = {LD_PER:.6f}   (t_i = i T/316, i = 1..316; NOT on the clock grid)")
print(f"log det J_sod(u_unif) = {LD_SOD:.6f}   (weights wk_d20_10000.npy, mc837.py, R = 10000, seed 20261006)")
print(f"SoD weights: sum = {WK_SOD.sum():.6f}, min = {WK_SOD.min():.3e}, max = {WK_SOD.max():.6f} (k = {int(WK_SOD.argmax())}), "
      f"#(w > 1) = {int((WK_SOD > 1).sum())}, #(w > 0.5) = {int((WK_SOD > 0.5).sum())}")
print(f"  -> the SoD weight vector is feasible for (B) (0 <= w_k <= 1, sum = W_B = S(u_unif)), so ld(J_sod) <= bound (B).")
print(f"  -> it is NOT feasible for (A): sum w = {WK_SOD.sum():.3f} > 316.")
kper = np.rint(ti/dt).astype(int)-1                       # nearest clock index of each periodic time (t = (k+1) dt)
assert np.unique(kper).size == Nper and kper.min() >= 0 and kper.max() < K
xi_per = np.zeros(K); xi_per[kper] = 1.0
ld_per_grid = ld(H_u, xi_per)
print(f"grid-rounded periodic schedule (316 distinct grid points, max |t_i - t_k| = {np.abs(ti - t[kper]).max()*1e3:.3f} ms): "
      f"log det = {ld_per_grid:.6f} (diff to J_per {ld_per_grid-LD_PER:+.6f}); feasible for (A)")
ld_U = {Wv: ld(H_u, np.full(K, Wv/K)) for Wv in (W_A, W_B)}
for Wv, v in ld_U.items(): print(f"uniform weights W/K on the grid, W = {Wv:.6f}: log det = {v:.6f} (diff to J_per {v-LD_PER:+.6f})")

# --- solve the three variants ---------------------------------------------------------------------------------------------
exchange(H_u[::25], 50.0, 1.0, tol=1e-3, max_sweeps=5)    # numba compile (not timed)
res = {}; XI = {}
for v, (Wv, cap) in VARIANTS.items():
    print(f"\n== variant ({v}): W = {Wv:.6f}, cap = {cap} ==")
    t0 = time.time(); rx = exchange(H_u, Wv, cap, tol=1e-10, batch=256); tx = time.time()-t0
    f_x, d_x = ld_grad(H_u, rx['xi']); kk = kkt_summary(rx['xi'], d_x, cap)
    print(f"[exchange, own] cold start (uniform weights): {tx:.2f} s, {rx['sweeps']} sweeps, {rx['n_ex']} exchanges")
    cx = certificate_full(H_u, rx['xi'], Wv, cap)
    print(f"   F(xi~) = {cx['F']:.10f}; cutting-plane upper bound U = {cx['U_cp']:.10f}, gap {cx['gap_cp']:.2e} nat; "
          f"duality bound {cx['U_dual']:.10f}, gap {cx['gap_dual']:.2e}; xi~^T d = {cx['xid']:.12f} (= 9)")
    print(f"   KKT: nu = {kk['nu']:.10f}, max violation = {kk['kkt_viol']:.2e} ({kk['kkt_viol_rel']:.2e} relative); "
          f"interior (0 < xi < cap) = {kk['n_interior']}, at cap = {kk['n_full']}, support = {kk['n_support']}, "
          f"#(1e-6 < xi < cap - 1e-6) = {int(((rx['xi'] > 1e-6) & (rx['xi'] < min(cap, 1e300)-1e-6)).sum())}")
    t0 = time.time(); rs = spg(H_u, Wv, cap, tol=1e-8, maxit=20000); ts = time.time()-t0
    cs = certificate_full(H_u, rs['xi'], Wv, cap)
    print(f"[SPG, own] cold start: {ts:.2f} s, {rs['it']} iterations ({rs['nfg']} f/grad evaluations); F = {cs['F']:.10f}, "
          f"U = {cs['U_cp']:.10f}, gap {cs['gap_cp']:.2e}; |F - F_exchange| = {abs(cs['F']-cx['F']):.2e}; "
          f"max|xi - xi_exchange| = {np.abs(rs['xi']-rx['xi']).max():.2e}")
    cv = {}
    for solver, kw in [('CLARABEL', {}), ('SCS', dict(eps=1e-9, max_iters=200000))]:
        try:
            r = cvx_solve(H_u, Wv, cap, solver, **kw)
            cc = certificate_full(H_u, r['xi'], Wv, cap)
            cv[solver] = dict(status=r['status'], time=r['time'], obj=r['obj'], F=cc['F'], U=cc['U_cp'], gap=cc['gap_cp'], proj=r['proj_dist'],
                              dxi=float(np.abs(r['xi']-rx['xi']).max()))
            print(f"[cvxpy {solver}{' ' + str(kw) if kw else ''}] status {r['status']}, {r['time']:.2f} s; solver objective {r['obj']:.10f}; "
                  f"after projection (moved {r['proj_dist']:.1e}): F = {cc['F']:.10f}, U = {cc['U_cp']:.10f}, gap {cc['gap_cp']:.2e}; "
                  f"|F - F_exchange| = {abs(cc['F']-cx['F']):.2e}; max|xi - xi_exchange| = {cv[solver]['dxi']:.2e}")
            XI[f'{v}_{solver}'] = r['xi']
        except Exception as e:
            cv[solver] = dict(status='FAILED', error=str(e)[:200])
            print(f"[cvxpy {solver}] FAILED: {str(e)[:120]}")
    XI[v] = rx['xi']; XI[f'{v}_spg'] = rs['xi']; XI[f'd_{v}'] = d_x
    res[v] = dict(W=Wv, cap=cap, F=cx['F'], U=cx['U_cp'], gap=cx['gap_cp'], U_dual=cx['U_dual'], time_exchange=tx, sweeps=rx['sweeps'],
                  kkt=kk, spg=dict(F=cs['F'], U=cs['U_cp'], gap=cs['gap_cp'], it=rs['it'], time=ts), cvx=cv)
    sup = rx['xi'] > 1e-9
    print(f"   |sum xi - W| = {abs(rx['xi'].sum()-Wv):.1e}; min positive weight = {rx['xi'][rx['xi'] > 0].min():.3e}; #(0 < xi <= 1e-9) = {int(((rx['xi'] > 0) & (rx['xi'] <= 1e-9)).sum())}")
    if np.isfinite(cap):
        bl = blocks(sup, t)
        print(f"   support blocks (contiguous runs of xi > 0): {len(bl)}; lengths (points) min/median/max = "
              f"{min(b[2] for b in bl)}/{int(np.median([b[2] for b in bl]))}/{max(b[2] for b in bl)}")
        print("   blocks [t_start, t_end] s (points; weight): " + '; '.join(
            f"[{a_:.3f},{b_:.3f}] ({n_}; {rx['xi'][(t >= a_-1e-12) & (t <= b_+1e-12)].sum():.2f})" for a_, b_, n_ in bl))
        res[v]['blocks'] = bl
    else:
        ks = np.flatnonzero(sup)
        print(f"   support points: {ks.size} (Caratheodory limit 45); max weight {rx['xi'].max():.4f}")
        print("   (t s, weight): " + '; '.join(f"({t[k]:.3f}, {rx['xi'][k]:.4f})" for k in ks))
        res[v]['support'] = [(float(t[k]), float(rx['xi'][k])) for k in ks]
    for lo, hi in [(0, 0.5), (0.5, 4.5), (4.5, 5.0)]:
        m_ = (t > lo) & (t <= hi)
        print(f"   weight in ({lo}, {hi}] s: {rx['xi'][m_].sum():.3f} ({100*rx['xi'][m_].sum()/Wv:.1f}%); "
              f"SoD weights there: {WK_SOD[m_].sum():.3f} ({100*WK_SOD[m_].sum()/WK_SOD.sum():.1f}%); window share {100*(hi-lo)/T:.0f}%")

# --- checks ---------------------------------------------------------------------------------------------------------------
print("\n== checks ==")
print(f"U_A >= grid-rounded periodic: {res['A']['U']:.6f} >= {ld_per_grid:.6f}: {res['A']['U'] >= ld_per_grid}")
print(f"U_B >= J_sod (SoD weights feasible for B): {res['B']['U']:.6f} >= {LD_SOD:.6f}: {res['B']['U'] >= LD_SOD}")
print(f"U_C >= F_A (C relaxes A): {res['C']['U']:.6f} >= {res['A']['F']:.6f}: {res['C']['U'] >= res['A']['F']}")
print(f"(C) support <= 45: {len(res['C']['support'])}")

# --- equivalent sample counts ---------------------------------------------------------------------------------------------
print("\n== equivalent total weight (how many full samples would match J_sod?) ==")
# uniform weights scale exactly: ld_U(W) = ld_U(316) + 9 log(W/316)
W_eq_U = W_A*np.exp((LD_SOD-ld_U[W_A])/P)
print(f"uniform placement on the grid: ld_U(W) = ld_U(316) + 9 log(W/316) = J_sod at W = {W_eq_U:.2f}")
_xw = [XI['A'].copy()]
def _A_of_W(Wv):
    r = exchange(H_u, Wv, 1.0, xi0=_xw[0]*Wv/_xw[0].sum(), tol=1e-9, batch=256); _xw[0] = r['xi']; return r['ub']-LD_SOD
t0 = time.time(); W_eq_A = brentq(_A_of_W, W_A, W_B, xtol=1e-4); tq = time.time()-t0
print(f"optimal placement with cap 1 (variant-A problem at total W, upper value U): U(W) = J_sod at W = {W_eq_A:.3f} ({tq:.1f} s, warm-started brentq, xtol 1e-4, inner gap <= 1e-9)")
print(f"  i.e. send-on-delta's 316.9 sends carry the information of {W_eq_A:.1f} optimally placed full samples on the grid "
      f"({W_eq_U:.1f} uniformly placed ones)")

# --- table ----------------------------------------------------------------------------------------------------------------
print("\n== Table 1: log det at the uniform input (nat); bound rows quote the upper value U; difference to J_sod(u_unif) and to J_per(316); axis ratio e^{-diff/18} ==")
rows = [('J_per(316)', LD_PER), ('grid-rounded periodic (316)', ld_per_grid), ('uniform weights W=316', ld_U[W_A]),
        (f'uniform weights W=S={W_B:.3f}', ld_U[W_B]), ('J_sod(u_unif)', LD_SOD),
        ('U_A (W=316, cap 1)', res['A']['U']), (f'U_B (W=S={W_B:.3f}, cap 1)', res['B']['U']), ('U_C (W=316, no cap)', res['C']['U'])]
print(f"{'row':34s} {'log det':>12s} {'- J_sod':>9s} {'ratio':>7s} {'- J_per':>9s} {'ratio':>7s}")
for nm, v in rows:
    print(f"{nm:34s} {v:12.6f} {v-LD_SOD:+9.4f} {axis_ratio(v-LD_SOD):7.4f} {v-LD_PER:+9.4f} {axis_ratio(v-LD_PER):7.4f}")
print("ratio = geometric-mean axis ratio of the 95% confidence ellipsoids, row vs reference (< 1: row is more informative).")
print(f"U_C - U_A = {res['C']['U']-res['A']['U']:+.6f} nat (price of the cap at W = 316)")
print(f"U_B - U_A = {res['B']['U']-res['A']['U']:+.6f} nat; 9 log(S/316) = {P*np.log(W_B/W_A):.6f}")
print(f"free-placement relaxation gap at equal total weight (uniform input): ld J_sod - U_B = {LD_SOD-res['B']['U']:+.6f} nat")
_r466 = exchange(H_u, 466.0, 1.0, xi0=XI['B']*466.0/W_B, tol=1e-10, batch=256)
print(f"note: with W = 466 (the rounded total of the original task) instead of S(u_unif) = {W_B:.6f}: U = {_r466['ub']:.6f} "
      f"({_r466['ub']-res['B']['U']:+.6f}); not used further")

print("\nStatement. U_A is the relaxation bound for 316 full measurements on the clock grid, each instant at most once: it bounds"
      " every scheme whose per-step information weights are each <= 1 and sum to <= 316. Send-on-delta is not such a scheme: with"
      f" 316.9 sends its weights sum to {WK_SOD.sum():.2f}, because the non-send steps carry censoring information. U_A is"
      " therefore not a limit for send-on-delta. U_B (weights each <= 1, total equal to send-on-delta's own S(u)) is the"
      " relaxation that does contain send-on-delta's weight pattern at this input.")

# --- figure ---------------------------------------------------------------------------------------------------------------
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(4, 1, figsize=(10, 8.2), sharex=True, gridspec_kw=dict(height_ratios=[1.1, 1, 1, 1]))
ax[0].plot(t, x, color=INK, lw=1.0); ax[0].set_ylabel('$x_k$')
ax[0].set_title('Uniform input: noise-free output, send-on-delta weights, and the optimal weights of relaxation (A) (316 full measurements)', color=INK, loc='left', fontsize=10)
ax[1].plot(t, WK_SOD, color=C1, lw=0.7); ax[1].set_ylabel(r'$\widehat w_k$ (SoD)')
ax[1].set_title(f'sum = {WK_SOD.sum():.2f} (316.9 sends)', loc='right', color=INK2, fontsize=8.5)
ax[2].fill_between(t, 0, XI['A'], step='mid', color=C2, lw=0); ax[2].set_ylabel(r'$\xi_k$ (A)'); ax[2].set_ylim(0, 1.08)
ax[2].set_title(f"sum = 316: {res['A']['kkt']['n_full']} at 1, {res['A']['kkt']['n_interior']} interior; "
                f"U_A = {res['A']['U']:.3f}", loc='right', color=INK2, fontsize=8.5)
ax[3].plot(t, XI['d_A']/res['A']['kkt']['nu'], color=C3, lw=0.9); ax[3].axhline(1.0, color=INK2, lw=0.8, ls='--')
ax[3].set_ylabel(r'$d_k/\nu$ (A)'); ax[3].set_xlabel('t (s)')
ax[3].set_title(r'KKT: $\xi_k=1$ where $d_k>\nu$, $\xi_k=0$ where $d_k<\nu$, interior only where $d_k=\nu$', loc='right', color=INK2, fontsize=8.5)
for a_ in ax: a_.grid(True, color=GRID, lw=0.6); a_.set_xlim(0, T)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_inner_A.png', dpi=150); plt.close(fig)
print(f"\nfigure: {OUT}/fig_inner_A.png")

json.dump(dict(LD_PER=LD_PER, LD_SOD=LD_SOD, S_sod=float(WK_SOD.sum()), ld_per_grid=ld_per_grid, ld_U={str(k): v for k, v in ld_U.items()},
               W_eq_U=float(W_eq_U), W_eq_A=float(W_eq_A), W_B=W_B, U_B466=float(_r466['ub']), res=res), open(f'{OUT}/inner_unif.json', 'w'), indent=1, default=float)
np.savez_compressed(f'{OUT}/inner_unif.npz', t=t, x=x, wk=WK_SOD, **XI)
log_runtime('bound_inner', T0, 'variants A, B (W = S(u_unif)), C at the uniform input; exchange + SPG + Clarabel + SCS; cutting-plane certificates; equivalent W by brentq')
