"""8.3.2: exact search of Delta_* = max{Delta : N^nf(Delta) >= N0} over critical values.
Needs mc83.py (setup, sod) and the compiled C routine ./sod (from sod.c: gcc -O2 -o sod sod.c -lm)."""
import numpy as np, subprocess, sys
exec(open('mc83.py').read().split('# --- noise-free')[0])   # x, K, T, sod(), etc.
N0 = int(sys.argv[1]) if len(sys.argv) > 1 else 320
lo = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0023   # lower cut-off (only limits work)
dx = np.diff(x); VT = np.abs(dx).sum(); U = VT/(N0-1)         # Lemma 6: feasible set within (0, U]
x.astype(np.float64).tofile('x.bin')
d = np.abs(x[:,None]-x[None,:]); iu = np.triu_indices(K,1); vals = d[iu]   # all |x_k - x_j|, j<k
cand = np.unique(vals[(vals>=lo)&(vals<=U)])[::-1]          # distinct critical values in [lo,U], descending
cand.tofile('cand.bin')
out = subprocess.run(['./sod','x.bin',str(K),'cand.bin',str(cand.size),str(N0)],capture_output=True,text=True).stdout.split()
Dstar = float(out[0]); n = int(out[1])
print(f"N0={N0}: V_T={VT!r}, U={U!r}, candidates={cand.size}, stopped at #{out[2]}")
print(f"Delta_* = {Dstar!r}  N^nf = {n}")
jk = np.argwhere(np.isclose(d, Dstar, rtol=0, atol=1e-18))[0] + 1   # 1-based (j,k) with |x_k - x_j| = Delta_*
print("critical pair (1-based):", sorted(jk))
