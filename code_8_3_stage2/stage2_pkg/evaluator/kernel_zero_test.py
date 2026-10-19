"""Speed-up Step 0 (spec reports_CN/plan_speedup_CN.md v3.1, §2 lemma 1): exact-zero test of the kernel expression as the
accepted evaluator computes it, i.e. the compiled numba function density_recursion_v1._kernel itself (same function, same
@njit(cache=True) flags: no fastmath, no parallel, python error model), not a re-implementation.

How a single kernel value is read exactly: _kernel(z, x, sigma, dz, gh) evaluates g(z_j) = exp(-u^2/2)/(sigma sqrt(2 pi)),
u = (z_j - x)/sigma, sums them into tot, normalises gh_j = g_j/tot and returns tot*dz. Called with a one-node grid and
dz = 1.0 it returns g itself (tot = 0.0 + g, times 1.0, both exact). If g == 0.0 then tot == 0.0 and the normalisation
raises ZeroDivisionError (numba's python error model), so the exception is raised exactly when the kernel value is 0.0.

Positions: |z - x| = 38.5, 39, 40, 50 sigma, sigma = Delta*/20, on both sides of every x_k (k = 1..K) of the four accepted
inputs U, D, E, N (z = x_k +- d sigma). Pass: every value at 39, 40, 50 sigma is exactly 0.0; the 38.5 sigma values are logged.
Writes logs/kernel_zero_test.log and, on pass, config_fast.json (rho = 39 sigma). Usage: python3 kernel_zero_test.py"""
import os, sys, json
import numpy as np
import density_recursion_v1 as dr

DIST = [38.5, 39.0, 40.0, 50.0]; TESTED = [39.0, 40.0, 50.0]


def kernel_value(z, x):
    """value of the accepted kernel expression at node z for centre x (exact; see module docstring)."""
    gh = np.empty(1)
    try: return dr._kernel(np.array([z]), x, dr.s, 1.0, gh)
    except ZeroDivisionError: return 0.0


if __name__ == '__main__':
    sys.stdout = dr.Tee(os.path.join(dr.LOGS, 'kernel_zero_test.log'))
    print(f"=== kernel_zero_test.py (speed-up Step 0, spec §2 lemma 1); kernel = density_recursion_v1._kernel "
          f"(numba targetoptions {dr._kernel.targetoptions}); sigma = Delta*/20 = {dr.s!r}; hardware {dr.CFG['hardware']}")
    allzero = {d: True for d in TESTED}; v385 = []
    for name in ['U', 'D', 'E', 'N']:
        xk, _ = dr.input_arrays(name)
        for d in DIST:
            vals = np.array([kernel_value(x+sg*d*dr.s, x) for x in xk for sg in (-1.0, 1.0)])
            nz = int((vals != 0.0).sum())
            line = f"KZT {name} |z-x| = {d} sigma: {vals.size} evaluations (both sides of {xk.size} x_k), non-zero {nz}"
            if nz: line += f", min {float(vals[vals != 0].min())!r}, max {float(vals.max())!r}"
            print(line)
            if d in TESTED and nz: allzero[d] = False
            if d == 38.5: v385.append(vals)
    v385 = np.concatenate(v385)
    print(f"KZT value at 38.5 sigma over all four inputs: min {float(v385.min())!r}, max {float(v385.max())!r}, exactly zero {int((v385 == 0).sum())} of {v385.size}")
    for d in TESTED: print(f"KZT all values exactly 0.0 at {d} sigma: {allzero[d]}")
    if allzero[39.0]:
        print("KZT PASS: rho = 39 sigma (spec §2)")
        cfg = dict(spec=dict(file='reports_CN/plan_speedup_CN.md', version='v3.1 (frozen)', commit='d4db8a6'),
                   predictions_commit='adcdb4b',
                   rho=dict(value=39*dr.s, definition='39 sigma', source='logs/kernel_zero_test.log (kernel exactly 0.0 at 39, 40, 50 sigma)'),
                   hardware=dr.CFG['hardware'])
        json.dump(cfg, open(os.path.join(dr.HERE, 'config_fast.json'), 'w'), indent=1)
    else:
        first = next((d for d in TESTED if allzero[d]), None)
        print(f"KZT FAIL at 39 sigma; first tested position giving exactly zero: {first if first else 'not found'}; STOP")
        sys.exit(1)
