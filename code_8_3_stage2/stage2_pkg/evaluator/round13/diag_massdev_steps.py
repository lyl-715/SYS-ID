"""Round 13 diagnostic for DECISION 13-30 (not a result): profile of the signed mass deviation dev_k = sum m - 1 after step k of the sigma/24
forward of one DP table, run with the conservation tolerance relaxed to 1e-10 for this diagnostic only.

Reports dev_k at k = 1000, 2000, 3000, 4000, 5000, the step of max |dev_k|, and dev_k around the input's short-window step (DECISION
13-22), with the increments D_k = dev_k - dev_{k-1}. Jump criterion (fixed before looking at the numbers): N "jumps" at its short-window
step ks if |D_ks| is the largest |D_k| of the run, or exceeds 10 x the median |D_k| over the 100 steps around ks (ks - 50 .. ks + 50).
Step numbering as in the forward's assertion message: dev_k is fw['mass'][k - 1] - 1 (k = 2 .. 5000; index 0 = m_2 at initialisation).

Usage: R13_INPUT=N python3 diag_massdev_steps.py table <path.npy>
       R13_INPUT=U python3 diag_massdev_steps.py regen <scan json> <run>   (regenerates the run's table by one DP at its lambda, checks the
                                                                           record EN, logdet bitwise, then profiles it)"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import eval13 as e13
import dp13

def profile(tab, label, short_steps):
    fw = e13.forward(e13.Z, e13.DZ, e13.XK, e13.KA, e13.KB, tab.rows_full, e13.X_PAD, 1e-10)
    rec, J = e13.record(fw)
    dev = fw['mass']-1.0; D = np.diff(dev); ad = np.abs(dev)
    step = lambda k: float(dev[k-1])
    out = dict(label=label, input=e13.NAME, EN=rec['EN'], logdet=rec['logdet'], min_m=float(fw['minm'].min()),
               dev_at={k: step(k) for k in (1000, 2000, 3000, 4000, 5000)},
               peak_step=int(np.argmax(ad))+1, peak_dev=float(dev[np.argmax(ad)]), n_over_1e12=int((ad > 1e-12).sum()),
               median_abs_increment=float(np.median(np.abs(D))), max_abs_increment=float(np.abs(D).max()), step_of_max_increment=int(np.argmax(np.abs(D)))+2)
    jumps = []
    for ks in short_steps:
        i = ks-1                                        # index of dev_ks
        Dks = float(dev[i]-dev[i-1])
        loc = np.abs(D[max(i-1-50, 0):i-1+51]); med = float(np.median(loc))
        rank = int((np.abs(D) > abs(Dks)).sum())+1
        jump = bool(abs(Dks) >= np.abs(D).max() or abs(Dks) > 10*med)
        out.setdefault('short_window', []).append(dict(step=ks, dev_around={k: step(k) for k in range(ks-3, ks+4)}, increment=Dks,
                                                       local_median_abs_increment=med, rank_of_abs_increment=rank, n_increments=int(D.size), jump=jump))
        jumps.append(jump)
    out['jump_at_short_window'] = any(jumps)
    print(json.dumps(out), flush=True)
    return out

if __name__ == '__main__':
    mode = sys.argv[1]
    w = e13.KB-e13.KA+1; short = (np.flatnonzero(w[1:] != w[1:].max())+2).tolist()
    if mode == 'table':
        path = sys.argv[2]
        tab = e13.Table11(np.load(path, mmap_mode='r'), e13.edge_set(), os.path.basename(path))
        profile(tab, os.path.basename(path), short)
    else:
        st = json.load(open(sys.argv[2])); run = int(sys.argv[3]); r = [x for x in st['runs'] if x['run'] == run][0]
        dp13.warmup(); des = dp13.Design11(lambda m: None)
        rec_sod, J_sod, W_sod = des.sod_reference(); gam, gbar, g1 = des.gamma_from_J(J_sod)
        assert gbar == st['gbar'] and g1 == st['gamma1']
        codes = np.empty((e13.K-1, e13.G), np.uint16); dsum, nbad = des.dp(e13.edge_set(), gam, r['lam'], codes)
        del des
        tab = e13.Table11(codes, e13.edge_set(), f"{st['label']}_r{run:02d}")
        o = profile(tab, f"{st['label']} run {run} (regenerated)", short)
        print(json.dumps(dict(regenerated=f"{st['label']} run {run}", lam=r['lam'], dsum_bitwise=bool(dsum == r['dsum']),
                              EN_bitwise=bool(o['EN'] == r['EN']), logdet_bitwise=bool(o['logdet'] == r['logdet']), mass_dev_max_record=r['mass_dev_max'])), flush=True)
