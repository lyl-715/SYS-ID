"""Speed-up Step 5: steady-state runtime of the window path (spec reports_CN/plan_speedup_CN.md v3.1, §5 item 5).

Same hardware and method as the acceptance run: one process, one thread (OMP/OPENBLAS/MKL = 1), numba compiled before timing
(_warmup), the timed span is the accepted density_recursion_v1.evaluate timer (recursion + J assembly), which on the window
path includes the destination-buffer zeroing and the per-step assertion. Each of the 12 evaluations (U, D, E, N x n = 6, 12, 24,
margin 8 Delta) is timed 3 times, in 3 rounds over the 12 in a fixed order. Reported: per evaluation the median of 3; for n = 24
the four medians, their maximum and the maximum single run; medians for n = 6 and 12; search cost 20000 x the median at n = 24
(median of the four per-input medians; also with their maximum), from the measurement only. Any assertion stops the script.
Each timed run is checked (untimed) to give w_k bitwise equal to the regression run in results/fast/.
Writes logs/speedup_timing.log. Usage: python3 timing_fast.py"""
import os, sys, json, platform, subprocess
import numpy as np
import density_recursion_v1 as dr
import evaluator_fast as ef

if __name__ == '__main__':
    sys.stdout = dr.Tee(os.path.join(dr.LOGS, 'speedup_timing.log'))
    cpu = [l.split(':', 1)[1].strip() for l in subprocess.run(['lscpu'], capture_output=True, text=True).stdout.splitlines()
           if l.startswith('Model name')][0]
    print(f"=== timing_fast.py (speed-up Step 5); hardware now: cpu {cpu}, cores {os.cpu_count()}; acceptance run: "
          f"{dr.CFG['hardware']['cpu']}, {dr.CFG['hardware']['cores']} cores; same: {cpu == dr.CFG['hardware']['cpu'] and os.cpu_count() == dr.CFG['hardware']['cores']}; "
          f"threads OMP={os.environ.get('OMP_NUM_THREADS')} OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} MKL={os.environ.get('MKL_NUM_THREADS')}")
    ef._warmup()
    T = {(nm, n): [] for nm in ef.INPUTS for n in ef.GRIDS}
    for rnd in range(3):
        for nm in ef.INPUTS:
            for n in ef.GRIDS:
                try:
                    rec, J, r = ef.evaluate_fast(nm, n, 8)
                except ef.AssertionStop as e:
                    print(f"TIME STOP: {e}"); sys.exit(1)
                same = np.array_equal(r['W'], np.load(os.path.join(dr.RES, 'fast', f'w_B_{nm}_n{n}_m8.npy')))
                T[(nm, n)].append(rec['runtime_s'])
                print(f"TIME round {rnd+1} B/{nm}/n={n} runtime_s {rec['runtime_s']!r} (w_k bitwise equal to the regression run: {same}; "
                      f"assertion failures {int((~r['assert_ok']).sum())})")
                if not same or (~r['assert_ok']).any(): print("TIME STOP: result differs or assertion"); sys.exit(1)
    med = {k: float(np.median(v)) for k, v in T.items()}
    for (nm, n), v in T.items():
        print(f"TIME-MEDIAN B/{nm}/n={n} median of 3 {med[(nm, n)]!r} (runs {v})")
    for n in ef.GRIDS:
        ms = [med[(nm, n)] for nm in ef.INPUTS]
        print(f"TIME-N n={n}: per-input medians {dict(zip(ef.INPUTS, ms))}; median of the four {float(np.median(ms))!r}; "
              f"max of the four {max(ms)!r}; max single run {max(max(T[(nm, n)]) for nm in ef.INPUTS)!r}")
    m24 = float(np.median([med[(nm, 24)] for nm in ef.INPUTS])); x24 = max(med[(nm, 24)] for nm in ef.INPUTS)
    for lab, tv in (('median of the four per-input medians', m24), ('max of the four per-input medians', x24)):
        print(f"SEARCH-COST 20000 evaluations x {lab} at n=24 {tv!r} s = {20000*tv!r} s = {20000*tv/3600!r} h (serial, one core)")
    acc = json.load(open(os.path.join(dr.RES, 'analysis.json')))['runtime']
    for n in ef.GRIDS:
        print(f"TIME-RATIO n={n}: accepted full-grid B mean {float(np.mean(acc[f'B_{n}']))!r} s (logs/analysis.log RUNTIME) / "
              f"window median of the four {float(np.median([med[(nm, n)] for nm in ef.INPUTS]))!r} s = "
              f"{float(np.mean(acc[f'B_{n}'])/np.median([med[(nm, n)] for nm in ef.INPUTS]))!r}")
    json.dump({f'{nm}_{n}': v for (nm, n), v in T.items()}, open(os.path.join(dr.RES, 'fast', 'timing.json'), 'w'), indent=1)
