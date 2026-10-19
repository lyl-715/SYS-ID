"""Diagnostic for the runtime table (P7): where the steady-state time of one version-B evaluation goes.
cProfile of density_recursion_v1.evaluate('B', <input>, n) after the JIT warm-up; prints the functions with the largest
total (self) time. cProfile adds per-call overhead, so its total exceeds the uninstrumented runtime of run_all.log;
only the split between the parts is used. Usage: python3 profile_runtime.py N 24 > logs/profile_runtime.log"""
import sys, cProfile, pstats, io
import density_recursion_v1 as dr

name, n = sys.argv[1], int(sys.argv[2])
dr._warmup(); dr.evaluate('B', name, 6)                  # JIT and first-call costs outside the profile
pr = cProfile.Profile(); pr.enable(); rec, J, r = dr.evaluate('B', name, n); pr.disable()
print(f"PROFILE B/{name}/n={n}: evaluate() runtime under cProfile {rec['runtime_s']:.2f} s (uninstrumented value: logs/run_all.log)")
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(12); print(s.getvalue())

# the numba kernels are invisible to cProfile: time them standalone on the same grid, 4999 calls each (their cost is
# O(G) and does not depend on the values of m); the rest of run_ab is the numpy/scipy part (cfun, exact P, slices, dots)
import time, numpy as np
xk, _ = dr.input_arrays(name); z, dz = dr.make_grid(xk, n, 8*dr.D, 'B'); ng = len(z); M = 20*n
gh = np.empty(ng); m = np.empty(ng); mn = np.empty(ng); Cg = np.empty(ng+1); Cm = np.empty(ng+1)
dr._kernel(z, xk[0], dr.s, dz, gh); m[:] = gh
t0 = time.perf_counter()
for k in range(1, dr.K): dr._kernel(z, xk[k], dr.s, dz, gh)
t1 = time.perf_counter()
for k in range(1, dr.K): dr._ab_step(m, mn, gh, Cg, Cm, M, 0.5)
t2 = time.perf_counter()
print(f"PROFILE B/{name}/n={n}: G = {ng}; numba _kernel x 4999 {t1-t0:.2f} s; numba _ab_step x 4999 {t2-t1:.2f} s; "
      f"cfun (cProfile cumtime) {pstats.Stats(pr).stats[[k for k in pstats.Stats(pr).stats if k[2] == 'cfun'][0]][3]:.2f} s")
