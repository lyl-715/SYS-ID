"""Diagnostic for S3/S4 (not part of the timing): where the time of one window-path evaluation goes. cProfile of
evaluator_fast.evaluate_fast(<input>, n) after warm-up (cProfile adds per-call overhead, so only the split is used), plus the
two numba parts timed standalone over 4999 calls on the same windows. Usage: python3 profile_fast.py N 24 > logs/speedup_profile.log"""
import sys, time, cProfile, pstats, io
import numpy as np
import density_recursion_v1 as dr
import evaluator_fast as ef

name, n = sys.argv[1], int(sys.argv[2])
ef._warmup(); ef.evaluate_fast(name, 6)
pr = cProfile.Profile(); pr.enable(); rec, J, r = ef.evaluate_fast(name, n); pr.disable()
st = pstats.Stats(pr); cf = [v for k, v in st.stats.items() if k[2] == 'cfun'][0]
print(f"PROFILE-FAST B/{name}/n={n}: evaluate under cProfile {rec['runtime_s']:.2f} s; cfun cumulative {cf[3]:.2f} s over {cf[1]} calls")
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(8); print(s.getvalue())
xk, _ = dr.input_arrays(name); z, dz = dr.make_grid(xk, n, 8*dr.D, 'B'); ng = len(z); M = 20*n
a, b, _, _ = ef.windows(xk, z[0], dz, ng); m = np.zeros(ng); mn = np.zeros(ng); gh = np.zeros(ng)
Cg = np.empty(ng+1); Cm = np.empty(ng+1); g1 = np.empty(ng); dr._kernel(z, xk[0], dr.s, dz, g1); m[a[1]:b[1]+1] = g1[a[1]:b[1]+1]
t0 = time.perf_counter()
for k in range(1, dr.K):
    gh[a[k-1]:b[k-1]+1] = 0.0; dr._kernel(z[a[k]:b[k]+1], xk[k], dr.s, dz, gh[a[k]:b[k]+1])
t1 = time.perf_counter()
for k in range(1, dr.K):
    an, bn = (a[k+1], b[k+1]) if k < dr.K-1 else (a[k], b[k])
    ef._ab_step_w(m, mn, gh, Cg, Cm, M, 0.5, a[k], b[k], a[k-1], b[k-1], an, bn)
t2 = time.perf_counter()
print(f"PROFILE-FAST B/{name}/n={n}: window kernel (zeroing + accepted _kernel on the slice) x 4999 {t1-t0:.2f} s; "
      f"_ab_step_w x 4999 {t2-t1:.2f} s (standalone, uninstrumented)")
