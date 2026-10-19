"""Step 0: freeze config.json for the evaluator acceptance run (spec reports_CN/plan_evaluator_CN.md v4.1).
Reads the full-precision a_m, chi_m of the four test inputs from the stored JSON outputs (no hand transcription)
and records the source file/index and log line of each.  Run once from this directory: python3 make_config.py"""
import json, os, platform, subprocess, numpy as np, scipy, numba, llvmlite

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
Dstar = 0.002384694482635785


def rec(path, key, idx, ident):
    r = json.load(open(os.path.join(PKG, path)))[key][idx]
    assert r['id'] == ident, (path, idx, r['id'])
    return r


def entry(r, ident, src_params, src_log, note=None):
    a = [float(v) for v in r['a']]; chi = [float(v) for v in r['chi']]
    assert all(0 <= c < 2*np.pi for c in chi)
    e = dict(id=ident, a=a, chi=chi, p=[v*v for v in a], sum_a2=float(np.sum(np.square(a))),
             source_params=src_params, source_log=src_log)
    if note: e['note'] = note
    return e


rD = rec('stage2/final_batch1.json', 'records', 0, 'NM-none-P5514')
rE = rec('stage3/dense_R2000.json', 'records', 15, 'S3-0.7-P4055')
rN = rec('stage3/unconstrained.json', 'records', 0, 'U-L1657')
amp = json.load(open(os.path.join(PKG, 'amp_nm_P5514.json')))

cpu = [l.split(':', 1)[1].strip() for l in subprocess.run(['lscpu'], capture_output=True, text=True).stdout.splitlines()
       if l.startswith('Model name')][0]

cfg = dict(
    spec=dict(file='reports_CN/plan_evaluator_CN.md', version='v4.1 (frozen)', bundle_commit='4e84ebc'),
    scenario=dict(d=20, Delta_star=Dstar, sigma=Dstar/20, sigma_def='Delta_star/20', dt=1e-3, K=5000, T=5.0,
                  system='G(s) = 2/(1+s)^2', omega='[3*sqrt(2), 3*pi, 12, 12.3]', a0=1.0, sum_a2=1.0),
    conventions=dict(
        time_index='t_k = k*dt, k = 1..K (t_1 = dt, t_K = T); array index k-1 (mc83.py: t = np.arange(1,K+1)*dt)',
        output='x_k = G(0)*1 + sum_m a_m |G(j w_m)| cos(w_m t_k + chi_m + arg G(j w_m)); built by mc85_scan.output(a, chi mod 2pi) '
               '(the function stage2_common.evaluate_crn uses), bitwise equal to mc83.x for U',
        H_columns='[1, a_m cos psi_km, -a_m sin psi_km] for m = 1..4, psi_km = w_m t_k + chi_m; parameter order '
                  '[G(0), Re G(jw_1), Im G(jw_1), ..., Re G(jw_4), Im G(jw_4)]; built by mc85_scan.regressors(a, chi mod 2pi, t), '
                  'bitwise equal to mc83.H for U; recursion and MC use the same x_k, h_k arrays',
        chi_reduction='chi reduced mod 2pi before forming x and H (as stage2_common.evaluate_crn); all four inputs already lie in [0, 2pi)',
        trigger='as mc83.sod: send iff |z_k - r_k| >= Delta; k = 1 forced send; reference = noisy z_k at the last send',
        recursion_start='f_2 = g_1 (r_2 = z_1); A/B: m_{2,j} = ghat_{1,j}',
        EN_main='A, B: E[N] with the discrete Phat (spec §4, internally consistent); exact-P E[N] recorded as the zero-cost diagnostic. '
                'v0: E[N] with the exact P of the prototype (its only E[N]); v0 has no omega-form Phat, so its Phat-E[N] is "not applicable"',
        reference_matrix='J_C = U at the same version, grid and margin; Delta m_D = log det J(u) - log det J_C; '
                         'lambda_i = eigh(J(u), J_C) ascending; m_E = lambda_1'),
    inputs=dict(
        U=dict(id='UNIF', a=[0.5]*4, chi=[0.0]*4, p=[0.25]*4, sum_a2=1.0,
               source_params='spec §5 table row U; mc83.py:5 (a=np.full(4,0.5); chi=np.zeros(4))', source_log='spec §5'),
        D=entry(rD, 'NM-none-P5514', 'code_8_3_stage2/stage2_pkg/stage2/final_batch1.json records[0] (identical a, chi in stage2/pool_R2000.json records[40])',
                'stage2/stage2_final.log:4 (p=(0.252,0.251,0.252,0.245) chi=(5.140,5.338,1.097,4.207)); stage2/stage2_final.log:16 (dmD_C +0.697 (0.004))',
                note=f"amp_nm_P5514.json (cited in spec §5) holds name={amp['name']}, best_p={amp['best_p']} and no chi: it is not this point. "
                     "Parameters taken from final_batch1.json records[0] as instructed by the user on 2026-10-19."),
        E=entry(rE, 'S3-0.7-P4055', 'code_8_3_stage2/stage2_pkg/stage3/dense_R2000.json records[15] (identical in stage3/front_union_R2000.json front[14])',
                'stage3/stage3_dense.log:117 (representative of the m_E >= 0.7 floor: p=(0.261,0.259,0.229,0.251) chi=(4.256,6.172,5.267,5.252) '
                'dmD_C=+0.262(0.010) mE_C=0.705); also :126, :131'),
        N=entry(rN, 'U-L1657', 'code_8_3_stage2/stage2_pkg/stage3/unconstrained.json records[0]',
                'stage3/stage3_unconstrained.log:31 ("Best evaluated point of the unbudgeted problem (not an optimum): U-L1657"); '
                ':32 (p, chi), :33 (B=423.96, dmD_C=+2.101 (0.011))',
                note='the log calls it "best evaluated point (not an optimum)"; spec §5 calls it the unbudgeted optimum')),
    grids=dict(n=[6, 12, 24], h='sigma/n', M='Delta/h = 20 n (integer)', nodes='z_i = z_0 + i h, z_0 = min x - margin, '
               'z = np.arange(min x - margin, max x + margin + h, h) as in the prototype',
               n48='only by spec §6 (absolute-quantity p_obs off the version\'s predicted order by > 0.5, or sign change of d1, d2) '
                   'or §7 (B/n=24 fails the MC comparison and the diagnostic extrapolation passes); then the full grid-switch rule '
                   '(stability 24/48, W from J_48, U recomputed at n=48, report states the actual grid)',
               roundoff_d2='|d2| <= 1e-8 max(1, |Q(24)|): "order undeterminable", never a trigger',
               U_relative='Delta m_D(U) = 0, m_E(U) = 1 identically: p_obs "not applicable"'),
    margins=dict(main='8*Delta', tail_check='16*Delta, version B, n = 12, all four inputs (relative quantities need U at 16*Delta)'),
    versions=dict(
        v0='prototype density_recursion.py run(): exact P = ndtr(beta)-ndtr(alpha), F = cumsum(f)*h, F(s+-Delta) by slices F[sh:], F[:-sh] '
           'padded with F[-1] and 0; c_k by the prototype\'s inline formula; no normalisation',
        A='omega_ij = 1 if |i-j| < M else 0',
        B='omega_ij = 1 if |i-j| < M; 1/2 if |i-j| = M; else 0',
        common_AB='ghat_{k,j} = g_k(z_j)/sum_j g_k(z_j); Phat_{k,i} = sum_j ghat_{k,j} omega_ij; '
                  'm_{k+1,j} = m_{k,j} Phat_{k,j} + ghat_{k,j} sum_i m_{k,i} (1 - omega_ij); no renormalisation of m'),
    implementation=dict(
        v0='numpy, the prototype\'s arithmetic (regression reference only)',
        A_B='numba-jitted mass recursion, sliding-window sums by prefix sums, O(G) per step; c_k by mc83.cfun (numpy/scipy; the reviewed '
            'implementation is reused, not ported) evaluated on the index range where m_k != 0 (exact: m_k = 0 elsewhere); '
            'w_k = sum_i m_{k,i} c_k(z_i)',
        runtime='steady state: import and JIT compilation excluded (one warm-up call before timing); serial, one process, '
                'OMP/OPENBLAS/MKL threads = 1',
        runtime_pass_line='version B on the accepted grid (n = 24, or 48 if switched): steady-state single evaluation <= 15 s (numba line, spec §6, §9)'),
    hardware=dict(cpu=cpu, cores=os.cpu_count(), ram_GB=16, python=platform.python_version(), numpy=np.__version__,
                  scipy=scipy.__version__, numba=numba.__version__, llvmlite=llvmlite.__version__),
    mc=dict(seed=20261019, R=10000, B=20, batch_size=500,
            draws='Z = np.random.default_rng(20261019).standard_normal((10000, K)); batch b = rows 500b .. 500b+499; '
                  'trajectory z = x + sigma Z[i]; the same Z for all four inputs (common random numbers)',
            trigger='stage2_common.sod_batch (bitwise equal to mc83.sod, check_trigger run first)',
            weights='c_k = cfun(alpha, beta) at the pre-update reference, c_1 = 1, arithmetic of stage2_common.evaluate_crn',
            point_estimates='from the pooled weights of all 10000 trajectories: Jbar = sigma^-2 sum_k wbar_k h_k h_k^T; batches only for se',
            se='s_batch/sqrt(20)',
            old_U=dict(seed=20261006, R=10000, weights='code_8_3_stage2/stage2_pkg/wk_d20_10000.npy',
                       se_old_EN=dict(value=0.022, source='code_8_3_stage2/stage2_pkg/README.md:21 (B_hat = 316.87 +- 0.022 (se))'),
                       se_old_logdet=dict(value=0.003, source='reports_CN/section_8_3_single_CN.md:527 (report value, not a raw log line)'),
                       se_old_S='not used: the same report gives two values (section_8_3_single_CN.md:522 batch se 0.12; :562 +-0.25); '
                                'raw difference only')),
    acceptance=dict(formal_items=199, items='per input: E[N], S, log det J, tr(Dbar)/9, 45 upper-triangle elements of Dbar (49 x 4 = 196); '
                    'paired Delta m_D of D, E, N vs U (3)', bonferroni_count=202, threshold='t_{19, 1-0.05/404} = 4.496',
                    whitening='W = J_rec^{-1/2}, J_rec = version B, n = 24 (n = 48 if the input switched); D_b = W Jhat_b W^T - I',
                    m_E='display and grid-stability only, not a formal test'),
)
json.dump(cfg, open(os.path.join(HERE, 'config.json'), 'w'), indent=1)
print(json.dumps(cfg['inputs'], indent=1)); print(cfg['hardware'])
