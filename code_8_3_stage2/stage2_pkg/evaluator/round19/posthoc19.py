"""Round 19, post-hoc descriptive diagnostics (written AFTER the summary; no new simulation, no new estimate, no variant: only the
saved arrays results/main.npz and results/rows.csv are read). Usage: R13_INPUT=U python3 posthoc19.py -> logs/posthoc.log,
results/posthoc.json.

D1  stage-1 identifiability: log det and condition number of J_1 (stage-1 SoD steps); RMS prediction error of theta_tilde over the
    stage-1 steps (fitted) and over the stage-2 steps (extrapolated), per K1.
D2  time split of the information: log det of the stage-1 and stage-2 parts of J for the baseline (SoD), the control and the primary;
    share of the transmissions in stage 1.
D3  CR reference for a varying per-run information: -log det mean_r(J_cond,r^-1) next to log det mean_r(J_cond,r) and D-hat.
D4  the replications in which e_k reaches a clip bound or N_2 is abnormal, with N and the prediction error.
"""
import os, json, csv
import numpy as np
import two_stage19 as t
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'results')
P, K, s, HK = t.P, t.K, t.s, t.HK
out_f = open(os.path.join(HERE, 'logs', 'posthoc.log'), 'w')
def out(m=''): print(m, flush=True); out_f.write(m+'\n')

A = np.load(os.path.join(RES, 'main.npz'))
rows = {}
with open(os.path.join(RES, 'rows.csv')) as f:
    for r in csv.DictReader(f): rows.setdefault(r['config'], []).append(r)
for n in rows: rows[n].sort(key=lambda r: int(r['i']))
res = {}

def Jpart(W, a, b): return (HK[a:b]*W[a:b, None]).T@HK[a:b]/s**2

out("D1 stage-1 identifiability (post hoc, descriptive)")
out(f"{'config':>9} {'K1':>5} {'logdet J_1':>10} {'cond J_1':>10} {'pe_rms stage 1 (median)':>24} {'pe_rms stage 2 (median)':>24} {'P90 stage 2':>11}")
res['D1'] = {}
for n in ('K1_1250', 'primary', 'K1_3750'):
    K1 = int(rows[n][0]['K1']); W = A[f'{n}_W']; J1 = Jpart(W, 0, K1)
    d = (A[f'{n}_theta_tilde']-t.THETA_STAR)@HK.T/s                 # (R, K) prediction error / sigma
    p1 = np.sqrt((d[:, :K1]**2).mean(1)); p2 = np.sqrt((d[:, K1:]**2).mean(1))
    res['D1'][n] = dict(K1=K1, logdet_J1=float(t.logdet(J1)), cond_J1=float(np.linalg.cond(J1)), pe1_median=float(np.median(p1)),
                        pe2_median=float(np.median(p2)), pe2_p90=float(np.percentile(p2, 90)))
    r = res['D1'][n]
    out(f"{n:>9} {K1:>5} {r['logdet_J1']:>10.3f} {r['cond_J1']:>10.3e} {r['pe1_median']:>24.4f} {r['pe2_median']:>24.4f} {r['pe2_p90']:>11.4f}")
J = A['baseline_W']; Jf = t.J_of_weights(J)
out(f"  context: cond of the whole-record baseline J {np.linalg.cond(Jf):.3e}")

out("")
out("D2 time split of the information (post hoc): log det of the stage parts of J (steps 1..2500 and 2501..5000)")
res['D2'] = {}
for n in ('baseline', 'control', 'primary'):
    W = A[f'{n}_W']; J1 = Jpart(W, 0, 2500); J2 = Jpart(W, 2500, K)
    N1 = np.array([float(r['N_1']) for r in rows[n]]); N = np.array([float(r['N']) for r in rows[n]])
    res['D2'][n] = dict(logdet_J1=float(t.logdet(J1)), logdet_J2=float(t.logdet(J2)), logdet_J=float(t.logdet(J1+J2)),
                        share_tx_stage1=float(N1.mean()/N.mean()))
    r = res['D2'][n]
    out(f"  {n:>9}: log det J(1..2500) {r['logdet_J1']:.4f}, log det J(2501..5000) {r['logdet_J2']:.4f}, log det J {r['logdet_J']:.4f}; "
        f"share of transmissions in steps 1..2500 {r['share_tx_stage1']:.4f}")
b = res['D2']['baseline']; c = res['D2']['control']
out(f"  stage-2 information, control minus baseline: {c['logdet_J2']-b['logdet_J2']:+.4f} nat (rule with the true model and feedback vs SoD, "
    f"steps 2501..5000); half-record SoD parts vs 197.201 - 9 ln 2 = {t.I_SOD-9*np.log(2):.4f}")

out("")
out("D3 CR reference for a varying per-run information (post hoc): I = log det mean J_cond; I_inv = -log det mean(J_cond^-1); D-hat")
res['D3'] = {}
J19 = json.load(open(os.path.join(RES, 'round19.json')))
for n in ('primary', 'K1_1250', 'K1_3750', 'control', 'baseline'):
    Jc = A[f'{n}_Jc'].reshape(-1, P, P)
    I = float(t.logdet(Jc.mean(0))); Iinv = float(-t.logdet(np.linalg.inv(Jc).mean(0)))
    Dh = J19['configs'][n]['estimator']['all']['D_hat']
    Dacc = J19['configs'][n]['estimator'].get('final accepted', J19['configs'][n]['estimator']['all'])['D_hat']
    res['D3'][n] = dict(I=I, I_inv=Iinv, I_minus_I_inv=I-Iinv, D_hat_all=Dh, D_hat_final_accepted=Dacc)
    out(f"  {n:>9}: I {I:.4f}, I_inv {Iinv:.4f} (I - I_inv {I-Iinv:+.4f}); D-hat all {Dh:.4f} (I_inv - D-hat {Iinv-Dh:+.4f}), "
        f"final accepted {Dacc:.4f} (I_inv - D-hat {Iinv-Dacc:+.4f})")

out("")
out("D4 replications with e_k at a clip bound or an abnormal N_2 (primary, K1_3750, control; K1_1250 summarised)")
res['D4'] = {}
for n in ('primary', 'K1_3750', 'control', 'K1_1250'):
    rr = [r for r in rows[n] if r['u_hit_lo'] == '1' or r['u_hit_hi'] == '1' or r['abnormal'] == '1']
    res['D4'][n] = [dict(i=int(r['i']), N=int(r['N']), N_2=int(r['N_2']), plan2=float(r['plan2']), u_lo=int(r['u_hit_lo']),
                         u_hi=int(r['u_hit_hi']), abnormal=int(r['abnormal']), pe_rms=float(r['pe_rms']),
                         s1_converged=int(r['s1_converged'])) for r in rr]
    if n != 'K1_1250':
        for x in res['D4'][n]: out(f"  {n}: {x}")
    else:
        pe = np.array([float(r['pe_rms']) for r in rows[n]]); N = np.array([float(r['N']) for r in rows[n]])
        ab = np.array([r['abnormal'] == '1' for r in rows[n]]); hi = np.array([r['u_hit_hi'] == '1' for r in rows[n]])
        res['D4_K1_1250'] = dict(n_flagged=len(rr), spearman_pe_N=float(np.corrcoef(np.argsort(np.argsort(pe)), np.argsort(np.argsort(N)))[0, 1]),
                                 pe_median_over_B=float(np.median(pe[N > t.B])), pe_median_not_over_B=float(np.median(pe[N <= t.B])),
                                 n_not_over_B=int((N <= t.B).sum()), pe_max_not_over_B=float(pe[N <= t.B].max()),
                                 n_hi_and_abnormal=int((hi & ab).sum()))
        r = res['D4_K1_1250']
        out(f"  K1_1250: {len(rr)} flagged replications; Spearman(pe_rms, N) {r['spearman_pe_N']:+.3f}; median pe_rms with N > B "
            f"{r['pe_median_over_B']:.3f} vs N <= B {r['pe_median_not_over_B']:.3f} (n = {r['n_not_over_B']}, max {r['pe_max_not_over_B']:.3f}); "
            f"e_k at 4 and abnormal N_2: {r['n_hi_and_abnormal']}")
with open(os.path.join(RES, 'posthoc.json'), 'w') as f: json.dump(res, f, indent=1)
out("=== ROUND 19 posthoc done")
