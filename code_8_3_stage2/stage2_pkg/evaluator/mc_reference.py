"""Step 4: Monte Carlo reference for the evaluator acceptance (spec reports_CN/plan_evaluator_CN.md v4.1, §7).

One new seed, one job: Z = default_rng(20261019).standard_normal((10000, K)), R = 10000 = B 20 batches x 500 trajectories
(batch b = rows 500b..500b+499), the same Z for the four inputs U, D, E, N (common random numbers).
Trigger stage2_common.sod_batch (check_trigger() first: bitwise equal to mc83.sod); weights c_k = mc83.cfun at the
pre-update reference, c_1 = 1, with the arithmetic of stage2_common.evaluate_crn. This script keeps, in addition, the
per-batch weight sums (per-batch S) and the per-trajectory S; it then reruns evaluate_crn(a, chi, Z, nbatch=20) and
checks that counts, pooled J, the 20 batch matrices, w_k and x_k are bitwise identical.

Point estimates from the pooled weights of all 10000 trajectories (Jbar = sigma^-2 sum_k wbar_k h_k h_k^T, then log det,
eigenvalues); batches only for standard errors se = s_batch/sqrt(20). Paired Delta m_D (D, E, N vs U) and m_E per batch;
nonlinearity diagnostic Dnl = log det Jbar - mean_b log det Jhat_b and -Dnl/19. m_E and lambda_2 - lambda_1 for display only.
U against the old seed-20261006 run (wk_d20_10000.npy): se_diff = sqrt(se_old^2 + se_new^2) where an old se has a source.

Outputs: results/mc/mc_<input>.npz, logs/mc_reference.log (one line per number).
Usage (from this directory): nohup python3 mc_reference.py > logs/mc_reference.stdout 2>&1 &
"""
import os, sys, json, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np
import density_recursion_v1 as dr

HERE = dr.HERE; PKG = dr.PKG
_cwd = os.getcwd(); os.chdir(PKG)
exec(open('stage2_common.py').read())        # sod_batch, check_trigger, evaluate_crn, cfun, output, regressors (read-only)
os.chdir(_cwd)
OUT = os.path.join(HERE, 'logs')             # stage2_common's OUT redirected; nothing is written to stage2/
MCD = os.path.join(dr.RES, 'mc'); os.makedirs(MCD, exist_ok=True)
CM = dr.CFG['mc']; SEED, RN, NB = CM['seed'], CM['R'], CM['B']; BS = RN//NB
INPUTS = ['U', 'D', 'E', 'N']
assert BS == CM['batch_size'] == 500 and D == dr.D and s == dr.s


def mc_input(name):
    """MC for one input on the common draws Z (global, shared by fork)."""
    e = dr.CFG['inputs'][name]
    xc, Hc = dr.input_arrays(name)
    cnt = np.zeros(RN); Si = np.zeros(RN); Wsum = np.zeros(K); Wb = np.zeros((NB, K)); chunk = 200
    for c0 in range(0, RN, chunk):
        zz = xc+s*Z[c0:c0+chunk]; eps, r = sod_batch(zz, D)
        al = (r[:, 1:]-D-xc[1:])/s; be = (r[:, 1:]+D-xc[1:])/s; cev, cno = cfun(al, be); cc = cev+cno
        for j in range(zz.shape[0]):
            i = c0+j; cnt[i] = eps[j].sum()
            c = np.concatenate([[1.0], cc[j]]); Wsum += c; Wb[i//BS] += c; Si[i] = c.sum()
    wk = Wsum/RN; J = (Hc*wk[:, None]).T@Hc/s**2
    Jbat = np.array([(Hc*(Wb[b]/BS)[:, None]).T@Hc/s**2 for b in range(NB)])
    ref = evaluate_crn(e['a'], e['chi'], Z, nbatch=NB, extras=True)            # the reviewed MC evaluator
    same = dict(cnt=np.array_equal(ref['cnt'], cnt), J=np.array_equal(ref['J'], J), Jbat=np.array_equal(ref['Jbat'], Jbat),
                wk=np.array_equal(ref['wk'], wk), x=np.array_equal(ref['x'], xc))
    np.savez(os.path.join(MCD, f'mc_{name}.npz'), cnt=cnt, Si=Si, wk=wk, wb=Wb/BS, J=J, Jbat=Jbat)
    return name, same


def ld(M): sg, v = np.linalg.slogdet(M); assert sg > 0; return float(v)
def se(v): return float(np.std(v, ddof=1)/np.sqrt(len(v)))


if __name__ == '__main__':
    sys.stdout = dr.Tee(os.path.join(dr.LOGS, 'mc_reference.log'))
    t0 = time.time()
    print(f"=== mc_reference.py {time.strftime('%Y-%m-%d %H:%M:%S')}; seed {SEED}, R = {RN} = {NB} x {BS}, CRN over {INPUTS}; "
          f"hardware {dr.CFG['hardware']}")
    print(check_trigger(verbose=False))
    Z = np.random.default_rng(SEED).standard_normal((RN, K))
    import multiprocessing as mp
    with mp.get_context('fork').Pool(4) as pl: res = pl.map(mc_input, INPUTS, chunksize=1)
    for name, same in res:
        print(f"MC {name} bitwise_equal_to_evaluate_crn {same}")
        assert all(same.values()), (name, same)
    print(f"MC draws and evaluation wall {time.time()-t0:.1f} s")

    R_ = {nm: dict(np.load(os.path.join(MCD, f'mc_{nm}.npz'))) for nm in INPUTS}
    for nm in INPUTS:
        r = R_[nm]
        r['ldb'] = np.array([ld(Jb) for Jb in r['Jbat']])
        r['ENb'] = r['cnt'].reshape(NB, BS).mean(1); r['Sb'] = r['wb'].sum(1)
        out = dict(EN=float(r['cnt'].mean()), EN_se=se(r['ENb']), S=float(r['wk'].sum()), S_se=se(r['Sb']),
                   logdet=ld(r['J']), logdet_se=se(r['ldb']), logdet_batch_mean=float(r['ldb'].mean()))
        out['Dnl'] = out['logdet']-out['logdet_batch_mean']; out['minus_Dnl_over_19'] = -out['Dnl']/(NB-1)
        out['EN_traj_sd'] = float(r['cnt'].std(ddof=1)); out['EN_clt_se'] = float(out['EN_traj_sd']/np.sqrt(RN))
        out['S_clt_se'] = float(r['Si'].std(ddof=1)/np.sqrt(RN))
        for k, v in out.items(): print(f"MC {nm} {k} {v!r}")
        r['out'] = out
    U = R_['U']
    for nm in ['D', 'E', 'N']:
        r = R_[nm]
        lam = dr.eigh(r['J'], U['J'], eigvals_only=True)
        dmb = r['ldb']-U['ldb']
        mEb = np.array([dr.eigh(r['Jbat'][b], U['Jbat'][b], eigvals_only=True)[0] for b in range(NB)])
        out = dict(dmD=r['out']['logdet']-U['out']['logdet'], dmD_se_paired=se(dmb), dmD_batch_mean=float(dmb.mean()),
                   mE=float(lam[0]), mE_se_paired=se(mEb), mE_batch_mean=float(mEb.mean()), lam2_minus_lam1=float(lam[1]-lam[0]))
        for k, v in out.items(): print(f"MC {nm} {k} {v!r}")
        for i, v in enumerate(lam): print(f"MC {nm} lam{i+1} {float(v)!r}")
        r['out'].update(out); r['dmb'] = dmb; r['mEb'] = mEb
    np.savez(os.path.join(MCD, 'mc_batch_stats.npz'), **{f'{nm}_{k}': R_[nm][k] for nm in INPUTS for k in ('ldb', 'ENb', 'Sb')},
             **{f'{nm}_{k}': R_[nm][k] for nm in ['D', 'E', 'N'] for k in ('dmb', 'mEb')})
    json.dump({nm: R_[nm]['out'] for nm in INPUTS}, open(os.path.join(MCD, 'mc_summary.json'), 'w'), indent=1)

    # --- U against the old seed-20261006 run (two independent MC estimates) ---------------------------------------
    old = CM['old_U']; xU, HU = dr.input_arrays('U')
    wk_old = np.load(os.path.join(PKG, 'wk_d20_10000.npy')); J_old = (HU*wk_old[:, None]).T@HU/s**2
    o = U['out']
    rows = [('EN', o['EN'], o['EN_se'], 316.87, 'README.md:21 (B_hat = 316.87)', old['se_old_EN']['value'], old['se_old_EN']['source']),
            ('logdet', o['logdet'], o['logdet_se'], ld(J_old), 'wk_d20_10000.npy (this script)', old['se_old_logdet']['value'],
             old['se_old_logdet']['source']),
            ('S', o['S'], o['S_se'], float(wk_old.sum()), 'wk_d20_10000.npy (this script)', None, old['se_old_S'])]
    for q, new, se_new, oldv, osrc, se_old, sesrc in rows:
        d = new-oldv
        print(f"OLD-U {q} new {new!r} (se {se_new!r}) old {oldv!r} [{osrc}] diff {d!r}")
        if se_old is not None:
            sd = float(np.hypot(se_old, se_new))
            print(f"OLD-U {q} se_old {se_old} [{sesrc}] se_diff {sd!r} diff/se_diff {d/sd!r}")
        else:
            print(f"OLD-U {q} se_old none used [{sesrc}]; raw difference only; diff in units of the new se {d/se_new!r} "
                  f"(a scale, not a test statistic)")
    print(f"=== mc_reference.py finished, wall {time.time()-t0:.1f} s")
