"""Round 14 driver (prompt reports_CN/prompt_round14_one_parameter_rule_EN.md). Rounds 4-13 are imported read-only; no DP is run.

One process per (stage, input):   R13_INPUT=<name> python3 run_round14.py <stage>
  part1    (U)  §2: 1a pure offsets, 1b fit_rule(B) = R(37, 13) on sigma/24 and sigma/120, 1c Monte Carlo of the lattice rule, 1d Monte Carlo of
                the continuous rule, 1e context lines from the archives                                     -> results/part1.json
  budgets  (U)  §3 2a: B in {160, 240, B_nom, 400, 480}: fit_rule, matched constant SoD (common13.matched_sod), round 10's class-W optimum
                (ub.solve) and the constant prediction-centred threshold (prob.fg(q_uniform))            -> results/budgets.json
  input    (SCH, P5514, LO, HI, N)  §3 2b: fit_rule(B_nom); I_full, matched SoD, T* from round13/results/partA_<name>.json
                                                                                                           -> results/input_<name>.json
  summary  (any)  round14.json: every number, Q1-Q5, predictions, checks.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, platform, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
R10 = os.path.join(EV, 'round10'); R11 = os.path.join(EV, 'round11'); R13 = os.path.join(EV, 'round13')
sys.path.insert(0, HERE)
import numpy as np
import scipy
import numba
from scipy.stats import norm
import rule14 as r14
e13, c13 = r14.e13, r14.c13
s, K, G, B_NOM, NAME = r14.s, r14.K, r14.G, r14.B_NOM, r14.NAME
RES = os.path.join(HERE, 'results'); LOGS = os.path.join(HERE, 'logs')
for _d in (RES, LOGS): os.makedirs(_d, exist_ok=True)
Stop = c13.Stop; jdump = c13.jdump; relerr = c13.relerr
REL = 1e-9
BUDGETS = (160.0, 240.0, B_NOM, 400.0, 480.0)
INPUTS_2B = ('SCH', 'P5514', 'LO', 'HI', 'N')
I_SOD_NOM = 197.20097668224076                         # constant SoD at Delta* (E[N] = B), round4/results/mc.json["U/SoD"]["logdet_eval"]
E_CONT = (1.50, 1.54, 1.56, 1.58, 1.60, 1.64)          # §2 1d
# §2 1a / 1b reference values (outside computation, prompt §2)
REF_1A = {36: (354.48311118855713, 1500.472396458184, 208.1697508229414), 37: (326.06595206524514, 1420.267400502641, 207.67570557568936),
          38: (299.41680466967017, 1341.9805960911076, 207.1658140469431), 40: (251.1909482065696, 1191.7274066151504, 206.0980178306077)}
REF_1B = {24: (316.7251743087337, 1392.8421245551658, 207.5002944958199, 99.1267), 120: (316.4986983108747, 1392.4592724251975, 207.49781936247174, 99.1269)}


class Tee:
    def __init__(self, path): self.f = open(path, 'a')
    def __call__(self, m=''): print(m, flush=True); self.f.write(m+'\n'); self.f.flush()

def versions():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__, numba_threads=numba.get_num_threads(),
                TOL24=e13.TOL24, TOL120=e13.TOL120)

FROZEN = ('round4', 'round5', 'round6', 'round7', 'round8', 'round8b', 'round10', 'round11', 'round13')
FROZEN_FILES = ('config.json', 'density_recursion_v1.py', 'evaluator_fast.py')

def cleanup_frozen(out):
    """prompt §0: delete empty folders created under round4/ (and rounds 5-13); confirm git sees no change in rounds 4-13 and the frozen files."""
    removed = []
    for rd in FROZEN:
        root = os.path.join(EV, rd)
        if not os.path.isdir(root): continue
        for dp_, dns, fns in os.walk(root, topdown=False):
            if '__pycache__' in dp_: continue
            if not os.listdir(dp_): os.rmdir(dp_); removed.append(os.path.relpath(dp_, EV))
    st = subprocess.run(['git', 'status', '--porcelain', '--'] + [os.path.join(EV, r) for r in FROZEN] + [os.path.join(EV, f) for f in FROZEN_FILES],
                        capture_output=True, text=True, cwd=EV).stdout.strip()
    out(f"CLEANUP removed empty folders {removed}; git status on rounds 4-13 and the frozen files: {'clean' if not st else st}")
    return removed, st == ''

def recshort(rec):
    return {k: rec[k] for k in ('EN', 'EN_P', 'S', 'logdet', 'cond', 'mass_dev_max', 'min_m', 'G', 'f', 'wall_s', 'rule', 'support_lo_min', 'support_hi_max') if k in rec}


# --- part 1 -------------------------------------------------------------------------------------------------------------------
def stage_part1(out):
    if NAME != 'U': raise Stop("part1 runs for input U")
    path = os.path.join(RES, 'part1.json'); res = json.load(open(path)) if os.path.exists(path) else {}
    res['versions'] = versions(); jdir = os.path.join(RES, 'part1')
    def save(): jdump(res, path)
    # 1a
    if 'a' not in res:
        rows = {}; ok = True
        for n, ref in REF_1A.items():
            rec, J = r14.evaluate(r14.Rule(n, 20), 1)
            rel = dict(EN=relerr(rec['EN'], ref[0]), S=relerr(rec['S'], ref[1]), logdet=relerr(rec['logdet'], ref[2]))
            p = all(v <= REL for v in rel.values()); ok &= p
            rows[n] = dict(e=n/24, rec=recshort(rec), ref=dict(EN=ref[0], S=ref[1], logdet=ref[2]), rel=rel, pass_=p)
            out(f"1a R({n}, 20) e = n/24 = {n/24:.4f}: EN {rec['EN']!r} S {rec['S']!r} logdet {rec['logdet']!r}; rel {rel['EN']:.1e} / {rel['S']:.1e} / "
                f"{rel['logdet']:.1e}; mass dev {rec['mass_dev_max']:.2e} ({rec['wall_s']:.1f} s) -> {'PASS' if p else 'FAIL'}")
        res['a'] = dict(rows=rows, pass_=bool(ok)); save()
        if not ok: raise Stop("1a does not reproduce to 1e-9")
    # 1b
    if 'b' not in res:
        fit, rule, J24, J120 = r14.fit_rule(B_NOM, out)
        c13.save_J(J24, jdir, 'R37_13_s24'); c13.save_J(J120, jdir, 'R37_13_s120')
        chk = {}
        for g, rec in ((24, fit['rec24']), (120, fit['rec120'])):
            ref = REF_1B[g]
            chk[g] = dict(EN=relerr(rec['EN'], ref[0]), S=relerr(rec['S'], ref[1]), logdet=relerr(rec['logdet'], ref[2]), cond=abs(rec['cond']-ref[3]),
                          cond_value=rec['cond'], ref=dict(EN=ref[0], S=ref[1], logdet=ref[2], cond=ref[3]))
            chk[g]['pass_'] = bool(chk[g]['EN'] <= REL and chk[g]['S'] <= REL and chk[g]['logdet'] <= REL and chk[g]['cond'] <= 5e-5)
            out(f"1b R({fit['n']}, {fit['j']}) sigma/{g}: EN {rec['EN']!r} S {rec['S']!r} logdet {rec['logdet']!r} cond {rec['cond']!r}; rel {chk[g]['EN']:.1e} / "
                f"{chk[g]['S']:.1e} / {chk[g]['logdet']:.1e}, |cond - ref| {chk[g]['cond']:.1e} -> {'PASS' if chk[g]['pass_'] else 'FAIL'}")
        r14_ = fit['evaluated24'].get('R(37, 14)')
        out(f"1b R(37, 14) sigma/24 (expected infeasible): {r14_}")
        res['b'] = dict(fit={k: v for k, v in fit.items() if k not in ('rec24', 'rec120')}, rec24=recshort(fit['rec24']), rec120=recshort(fit['rec120']),
                        n_j_ok=bool((fit['n'], fit['j']) == (37, 13)), check=chk, R37_14=r14_,
                        pass_=bool((fit['n'], fit['j']) == (37, 13) and chk[24]['pass_'] and chk[120]['pass_']))
        save()
        if not res['b']['pass_']: raise Stop("1b does not reproduce")
    b = res['b']
    # 1c Monte Carlo of the lattice rule
    if 'c' not in res:
        rule = r14.Rule(b['fit']['n'], b['fit']['j'])
        mc = r14.monte_carlo('lattice', rule.n_p)
        r120 = b['rec120']
        mc.update(EN_eval120=r120['EN'], S_eval120=r120['S'], logdet_eval120=r120['logdet'],
                  z_EN=(r120['EN']-mc['EN_mc'])/mc['EN_se'], z_S=(r120['S']-mc['S_mc'])/mc['S_se'], z_logdet=(r120['logdet']-mc['logdet_mc'])/mc['logdet_batch_se'],
                  EN_eval24=b['rec24']['EN'], S_eval24=b['rec24']['S'], logdet_eval24=b['rec24']['logdet'])
        mc['pass_'] = bool(max(abs(mc['z_EN']), abs(mc['z_S']), abs(mc['z_logdet'])) <= 3 and mc['n_clipped'] == 0)
        out(f"1c MC lattice {rule.label()}: EN {mc['EN_mc']!r} ({mc['EN_se']:.4f}) S {mc['S_mc']!r} ({mc['S_se']:.3f}) logdet {mc['logdet_mc']!r} ({mc['logdet_batch_se']:.4f}); "
            f"z(sigma/120) EN {mc['z_EN']:+.2f} S {mc['z_S']:+.2f} logdet {mc['z_logdet']:+.2f}; clipped {mc['n_clipped']}; {mc['wall_s']:.1f} s -> {'PASS' if mc['pass_'] else 'FAIL'}")
        res['c'] = mc; save()
    # 1d Monte Carlo of the continuous rule
    if 'd' not in res:
        runs = {}
        for e in E_CONT:
            mc = r14.monte_carlo('cont', e); runs[f"{e:.2f}"] = mc
            out(f"1d MC continuous e = {e:.2f}: EN {mc['EN_mc']!r} ({mc['EN_se']:.4f}) S {mc['S_mc']!r} ({mc['S_se']:.3f}) logdet {mc['logdet_mc']!r} "
                f"({mc['logdet_batch_se']:.4f}); {mc['wall_s']:.1f} s")
        es = np.array(E_CONT); ens = np.array([runs[f"{e:.2f}"]['EN_mc'] for e in E_CONT]); lds = np.array([runs[f"{e:.2f}"]['logdet_mc'] for e in E_CONT])
        k = [i for i in range(len(es)-1) if (ens[i]-B_NOM)*(ens[i+1]-B_NOM) <= 0]
        if not k: raise Stop(f"1d: B not bracketed by the Monte Carlo E[N] {ens.tolist()}")
        i = k[0]; t = (ens[i]-B_NOM)/(ens[i]-ens[i+1])
        eC = float(es[i]+t*(es[i+1]-es[i])); ldC = float(lds[i]+t*(lds[i+1]-lds[i]))
        eU = float(np.round(np.ceil(eC/0.005-1e-9)*0.005, 3))
        mcU = r14.monte_carlo('cont', eU)
        lat = b['fit']
        d = dict(runs=runs, bracket=[float(es[i]), float(es[i+1])], e_C=eC, logdet_interp=ldC, e_up=eU, mc_up=mcU,
                 EN_minus_B=mcU['EN_mc']-B_NOM, z_EN_vs_B=(mcU['EN_mc']-B_NOM)/mcU['EN_se'], meets_budget=bool(mcU['EN_mc'] <= B_NOM),
                 dI_vs_SoD=mcU['logdet_mc']-I_SOD_NOM, lattice_e_eff=lat['e_eff'], e_C_minus_e_eff=eC-lat['e_eff'],
                 logdet_minus_lattice120=mcU['logdet_mc']-b['rec120']['logdet'], logdet_interp_minus_lattice120=ldC-b['rec120']['logdet'])
        d['Q2'] = bool(d['meets_budget'] and abs(d['logdet_minus_lattice120']) <= 0.05)
        out(f"1d interpolation: E[N] = B between e = {es[i]:.2f} and {es[i+1]:.2f} -> e_C = {eC!r}, interpolated logdet {ldC!r}; extra MC at e = {eU:.3f}: "
            f"EN {mcU['EN_mc']!r} ({mcU['EN_se']:.4f}; EN - B = {d['EN_minus_B']:+.4f}, z {d['z_EN_vs_B']:+.2f}) S {mcU['S_mc']!r} logdet {mcU['logdet_mc']!r} "
            f"({mcU['logdet_batch_se']:.4f}); dI vs constant SoD {d['dI_vs_SoD']!r}; lattice e_eff {lat['e_eff']!r} (e_C - e_eff {d['e_C_minus_e_eff']:+.4f}); "
            f"logdet - lattice sigma/120 {d['logdet_minus_lattice120']:+.4f}")
        res['d'] = d; save()
    # 1e context lines (archives) and derived
    r11b = json.load(open(os.path.join(R11, 'addendum', 'results', 'round11b.json')))
    Tp = r11b['Tprime']; UBp = r11b['proposition5']['UB']
    J_full, I_full = c13.full_information()
    J_sod = np.load(os.path.join(R13, 'results', 'gate_U_design', 'J_sod_reference_s24.npy'))
    I24, I120 = b['rec24']['logdet'], b['rec120']['logdet']
    e = dict(sources=dict(I_SoD='round4/results/mc.json["U/SoD"]["logdet_eval"] (as quoted in the prompt)', Tprime='round11/addendum/results/round11b.json Tprime',
                          certificate='round11/addendum/results/round11b.json proposition5.UB (B + 0.25)', J_SoD='round13/results/gate_U_design/J_sod_reference_s24.npy',
                          I_full='slogdet(HK^T HK / sigma^2)'),
             I_SoD=I_SOD_NOM, Tprime_I24=Tp['I24'], Tprime_I120=Tp['I120'], Tprime_EN120=Tp['EN120'], cert_Bp025=UBp, I_full=I_full,
             cond_J_full=float(np.linalg.cond(J_full)), cond_J_SoD=float(np.linalg.cond(J_sod)), I_SoD_from_J=float(np.linalg.slogdet(J_sod)[1]))
    for g, I, ITp in ((24, I24, Tp['I24']), (120, I120, Tp['I120'])):
        gap = UBp-I
        e[f'sigma{g}'] = dict(I_rule=I, dI_vs_SoD=I-I_SOD_NOM, loss_vs_Tprime=ITp-I, gap_to_cert=gap, deff_lower=float(np.exp(-gap/9)),
                              sd_ratio_SoD=float(np.exp((I_full-I_SOD_NOM)/18)), sd_ratio_rule=float(np.exp((I_full-I)/18)), sd_ratio_Tprime=float(np.exp((I_full-ITp)/18)))
    out(f"1e context: I_SoD {I_SOD_NOM!r}; T' {Tp['I24']!r} (sigma/24) / {Tp['I120']!r} (sigma/120), EN120 {Tp['EN120']!r}; certificate (B + 0.25) {UBp!r}; "
        f"I_full {I_full!r}; cond J_full {e['cond_J_full']:.4f}, cond J_SoD {e['cond_J_SoD']:.4f}")
    for g in (24, 120):
        x = e[f'sigma{g}']
        out(f"1e rule sigma/{g}: I {x['I_rule']!r}; dI vs SoD {x['dI_vs_SoD']:.4f}; loss vs T' {x['loss_vs_Tprime']:.4f}; gap to certificate {x['gap_to_cert']:.4f}; "
            f"exp(-gap/9) {x['deff_lower']:.4f}; sd ratio to full data SoD {x['sd_ratio_SoD']:.4f} rule {x['sd_ratio_rule']:.4f} T' {x['sd_ratio_Tprime']:.4f}")
    res['e'] = e; save()
    return res


# --- budgets (2a) -----------------------------------------------------------------------------------------------------------------
def stage_budgets(out):
    if NAME != 'U': raise Stop("budgets runs for input U")
    sys.path.insert(0, R10); import ub                      # round 10 (read-only)
    path = os.path.join(RES, 'budgets.json'); res = json.load(open(path)) if os.path.exists(path) else dict(rows={})
    res['versions'] = versions(); jdir = os.path.join(RES, 'budgets')
    def save(): jdump(res, path)
    p1 = json.load(open(os.path.join(RES, 'part1.json')))
    r10 = json.load(open(os.path.join(R10, 'results', 'round10.json')))['budgets']['B316']
    for Bt in BUDGETS:
        key = f"{Bt:.10g}"
        if key in res['rows'] and res['rows'][key].get('done'): out(f"BUDGET {key}: done, skipped"); continue
        out(f"BUDGET B = {Bt!r}")
        row = dict(B=Bt)
        fit, rule, J24, J120 = r14.fit_rule(Bt, out)
        c13.save_J(J24, jdir, f'rule_B{key}_s24'); c13.save_J(J120, jdir, f'rule_B{key}_s120')
        row['fit'] = {k: v for k, v in fit.items() if k not in ('rec24', 'rec120')}; row['rec24'] = recshort(fit['rec24']); row['rec120'] = recshort(fit['rec120'])
        if Bt == B_NOM:                                        # check 5: bitwise the rule of 1b
            b = p1['b']
            row['check5'] = dict(n_j=bool((fit['n'], fit['j']) == (b['fit']['n'], b['fit']['j'])),
                                 rec24_bitwise=bool(all(fit['rec24'][k] == b['rec24'][k] for k in ('EN', 'S', 'logdet'))),
                                 rec120_bitwise=bool(all(fit['rec120'][k] == b['rec120'][k] for k in ('EN', 'S', 'logdet'))))
            row['check5']['pass_'] = all(row['check5'].values())
            out(f"CHECK5 nominal budget: fitted rule vs 1b {row['check5']}")
        save()
        m, _, _ = c13.matched_sod(Bt, log=out, outdir=jdir, tag=f'sod_B{key}')
        row['matched'] = m; save()
        if not m['accepted']: out(f"BUDGET {key}: matched SoD not accepted (residual {m['residual120']!r}) - reported")
        # round 10: class-W optimum and the constant prediction-centred threshold
        prob = ub.Problem(e13.HK, s, ub.c_max, ub.dc_max)
        t0 = time.perf_counter(); sol = ub.solve(prob, Bt-1.0, tol=0.01); t10 = time.perf_counter()-t0
        qu = (Bt-1.0)/(K-1.0); f_u = ub.Problem(e13.HK, s, ub.c_max, ub.dc_max).fg(np.full(K-1, qu))[0]
        row['round10'] = dict(f=sol['f'], u_cert=sol['u_cert'], gap=sol['gap'], n_iter=sol['n_iter'], history0_f=sol['history'][0][1], f_uniform=f_u,
                              q_uniform=qu, a_uniform=float(norm.isf(qu/2)), wall_s=t10)
        if Bt == B_NOM:
            row['check4'] = dict(f=sol['f'], f_ref=r10['f'], f_rel=relerr(sol['f'], r10['f']), u_cert=sol['u_cert'], u_cert_ref=r10['u_cert'],
                                 u_cert_rel=relerr(sol['u_cert'], r10['u_cert']), history0_f=sol['history'][0][1], history0_ref_4dp=208.9566,
                                 history0_4dp_ok=bool(round(sol['history'][0][1], 4) == 208.9566), f_uniform_equals_history0=bool(f_u == sol['history'][0][1]))
            row['check4']['pass_'] = bool(row['check4']['f_rel'] <= REL and row['check4']['u_cert_rel'] <= REL and row['check4']['history0_4dp_ok'])
            out(f"CHECK4 round 10 at the nominal budget: f {sol['f']!r} (archive {r10['f']!r}, rel {row['check4']['f_rel']:.1e}); u_cert {sol['u_cert']!r} "
                f"(archive {r10['u_cert']!r}, rel {row['check4']['u_cert_rel']:.1e}); history[0].f {sol['history'][0][1]!r} (208.9566 at 4 dp: "
                f"{row['check4']['history0_4dp_ok']}) -> {'PASS' if row['check4']['pass_'] else 'FAIL'}")
        qs = fit['q_star']
        I_r, I_s = fit['rec120']['logdet'], m['logdet120']
        row['derived'] = dict(q_star=qs, e1=fit['e1'], e_eff=fit['e_eff'], delta=fit['delta'], EN120=fit['rec120']['EN'], I_rule=I_r, I_SoD=I_s, dI=I_r-I_s,
                              ips_rule=(fit['rec120']['S']-1)/(fit['rec120']['EN']-1), ips_SoD=(m['S120']-1)/(m['EN120']-1), I_const_threshold=f_u,
                              I_classW=sol['f'], u_cert=sol['u_cert'], classW_minus_rule=sol['f']-I_r, rule_minus_const_threshold=I_r-f_u,
                              I_rule24=fit['rec24']['logdet'], EN24=fit['rec24']['EN'])
        row['done'] = True; res['rows'][key] = row; save()
        d = row['derived']
        out(f"BUDGET {key} DONE: q* {qs:.5f} e1 {d['e1']:.4f} e_eff {d['e_eff']:.4f} delta {d['delta']:+.4f}; rule {rule.label()} EN120 {d['EN120']:.3f} I {I_r:.4f}; "
            f"SoD M* {m['M_star']} I {I_s:.4f} (residual {m['residual120']:+.3f}); dI {d['dI']:.4f}; info/send rule {d['ips_rule']:.4f} SoD {d['ips_SoD']:.4f}; "
            f"constant threshold {f_u:.4f}; class-W f {sol['f']:.4f} (u_cert {sol['u_cert']:.4f}); class-W - rule {d['classW_minus_rule']:.4f}")
    return res


# --- inputs (2b) ----------------------------------------------------------------------------------------------------------------
def stage_input(out):
    if NAME not in INPUTS_2B: raise Stop(f"input stage is for {INPUTS_2B}")
    pa = json.load(open(os.path.join(R13, 'results', f'partA_{NAME}.json')))
    if pa.get('status') != 'done': raise Stop(f"round 13 Part A result for {NAME} is not complete")
    fit, rule, J24, J120 = r14.fit_rule(B_NOM, out)
    jdir = os.path.join(RES, f'input_{NAME}'); c13.save_J(J24, jdir, 'rule_s24'); c13.save_J(J120, jdir, 'rule_s120')
    dv = pa['derived']; I_r = fit['rec120']['logdet']
    res = dict(input=NAME, versions=versions(), fit={k: v for k, v in fit.items() if k not in ('rec24', 'rec120')}, rec24=recshort(fit['rec24']), rec120=recshort(fit['rec120']),
               partA=dict(I_full=dv['I_full'], I_table=dv['I_table'], I_SoD=dv['I_SoD'], M_star=pa['matched']['M_star'], source=f'round13/results/partA_{NAME}.json derived'),
               derived=dict(e_eff=fit['e_eff'], e1=fit['e1'], delta=fit['delta'], I_rule=I_r, I_rule24=fit['rec24']['logdet'], c_rule=I_r-dv['I_full'],
                            rule_minus_table=I_r-dv['I_table'], rule_minus_SoD=I_r-dv['I_SoD'], EN120=fit['rec120']['EN'], S120=fit['rec120']['S']))
    jdump(res, os.path.join(RES, f'input_{NAME}.json'))
    d = res['derived']
    out(f"INPUT {NAME} DONE: {rule.label()} e_eff {d['e_eff']:.4f} delta {d['delta']:+.4f}; I_rule {I_r:.4f} (sigma/24 {d['I_rule24']:.4f}); c_rule {d['c_rule']:.4f}; "
        f"rule - table {d['rule_minus_table']:+.4f}; rule - SoD {d['rule_minus_SoD']:.4f}")
    return res


STAGES = ('part1', 'budgets', 'input', 'summary')

def main():
    numba.set_num_threads(int(os.environ.get('R13_NUMBA_THREADS', '4')))
    for stg in sys.argv[1:]:
        if stg not in STAGES: raise SystemExit(f"unknown stage {stg}")
        out = Tee(os.path.join(LOGS, f'{stg}_{NAME}.log' if stg != 'summary' else 'summary.log'))
        out(f"=== ROUND 14 stage {stg} input {NAME} started {time.strftime('%Y-%m-%d %H:%M:%S')}; python {platform.python_version()}, numpy {np.__version__}, "
            f"scipy {scipy.__version__}, numba {numba.__version__}; B = {B_NOM!r}; G = {G}; TOL24 {e13.TOL24:g}, TOL120 {e13.TOL120:g}")
        t0 = time.perf_counter()
        try:
            if stg == 'part1': stage_part1(out)
            elif stg == 'budgets': stage_budgets(out)
            elif stg == 'input': stage_input(out)
            elif stg == 'summary':
                import summary14; summary14.stage(out)
        except (Stop, e13.AssertionStop, AssertionError) as e:
            out(f"STOP {type(e).__name__}: {e}"); out(f"=== ROUND 14 stage {stg} input {NAME} STOPPED after {time.perf_counter()-t0:.1f} s")
            cleanup_frozen(out); raise
        out(f"=== ROUND 14 stage {stg} input {NAME} done in {time.perf_counter()-t0:.1f} s")
        cleanup_frozen(out)


if __name__ == '__main__':
    main()
