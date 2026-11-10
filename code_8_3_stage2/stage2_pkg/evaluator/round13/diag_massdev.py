"""Round 13 Part A diagnostic (not a result): per-step mass-conservation deviation |sum m - 1| of the sigma/24 forward of DP tables, run
with the conservation tolerance relaxed to 1e-10 for this diagnostic only (eval13.TOL24 = 1e-12 is unchanged and still applies to every
reported evaluation). Used for input N, whose scan j0 stopped at run 4 (|sum m - 1| = 1.004e-12 at step 4217 > 1e-12).
Usage: R13_INPUT=N ROUND13_WORK=<dir> python3 diag_massdev.py <table.npy> ..."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import eval13 as e13
res = []
for path in sys.argv[1:]:
    tab = e13.Table11(np.load(path, mmap_mode='r'), e13.edge_set(), os.path.basename(path))
    fw = e13.forward(e13.Z, e13.DZ, e13.XK, e13.KA, e13.KB, tab.rows_full, e13.X_PAD, 1e-10)
    dev = np.abs(fw['mass']-1.0); k = int(np.argmax(dev))
    rec, J = e13.record(fw)
    d = dict(table=os.path.basename(path), input=e13.NAME, G=e13.G, EN=rec['EN'], logdet=rec['logdet'], max_dev=float(dev.max()), step_of_max=k+1,
             n_steps_over_1e12=int((dev > 1e-12).sum()), n_steps_over_5e13=int((dev > 5e-13).sum()), first_step_over_1e12=(int(np.flatnonzero(dev > 1e-12)[0])+1 if (dev > 1e-12).any() else None),
             dev_at_step_4217=float(dev[4216]), min_m=float(fw['minm'].min()))
    res.append(d); print(json.dumps(d), flush=True)
