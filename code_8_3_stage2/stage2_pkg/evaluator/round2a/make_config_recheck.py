"""Round 2a, A0: freeze config_recheck.json before any evaluation (spec reports_CN/plan_recheck_bounds_CN.md v4.1, §A1-§A3).

The seven candidates (decided by the user on 2026-10-20, because the spec's "seven final candidates of the stage-4 frozen pool"
and its required members could not both hold): the stage-2 finalists that are in the stage-4 frozen pool, without UNIF (which is
the reference row), plus the unbudgeted U-L1657. Full a_m, chi_m are read from their JSON records (file and index recorded).
MC values: Delta m_D^(C) and its paired batch se (10 x 400) of the R = 4000 verification batch 1 (stage2_final.py, seed 20261013).
U-L1657 has no R = 4000 batch: its z column is "未定" (user decision); its R = 2000 value is recorded for reference only.
tau_N, tau_E: 10 x the largest |Q(12) - Q(24)| of the four acceptance inputs (evaluator/logs/analysis.log, STAB lines, version B),
rounded up to one significant figure. Budget (spec v4.2, user correction before any run): 8.3.6(g), relative to the uniform input,
B := E[N](u_unif) from the same evaluator (B, n = 24); the uniform input qualifies by definition.
Usage (from this directory): python3 make_config_recheck.py"""
import os, json, math
from decimal import Decimal, ROUND_CEILING

HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE); PKG = os.path.dirname(EV)


def ceil_1sig(x):
    """round up to one significant figure, in decimal (0.03784 -> 0.04)."""
    d = Decimal(repr(x)); e = d.adjusted()
    return float((d.scaleb(-e).to_integral_value(rounding=ROUND_CEILING)).scaleb(e))

def stab(input_, q):
    """the STAB line of analysis.log for input and quantity at 12/24: (line number, text, value)."""
    for i, l in enumerate(open(os.path.join(EV, 'logs', 'analysis.log')), 1):
        if l.startswith(f'STAB {input_} 12/24 {q} '): return i, l.strip(), float(l.split()[4])
    raise KeyError((input_, q))

def log_line(path, start):
    for i, l in enumerate(open(os.path.join(PKG, path)), 1):
        if l.startswith(start): return i, l.rstrip()
    raise KeyError((path, start))


fb = json.load(open(os.path.join(PKG, 'stage2/final_batch1.json')))
FLOOR = {'NM-none-P5514': None, 'NM-0.3-P4312': 0.3, 'AB-0.5': 0.5, 'NM-0.7-UNIF': 0.7, 'NM-none-P5450': None, 'P5014': None}
cands = []
for idx, r in enumerate(fb['records']):
    if r['id'] == 'UNIF': continue
    ln, txt = log_line('stage2/stage2_final.log', f"{r['id']} ")
    cands.append(dict(id=r['id'], floor_lambda0=FLOOR[r['id']], a=[float(v) for v in r['a']], chi=[float(v) for v in r['chi']],
                      p=[float(v)**2 for v in r['a']], source_params=f'code_8_3_stage2/stage2_pkg/stage2/final_batch1.json records[{idx}]',
                      mc=dict(dmD=float(r['dmD_C']), paired_se=float(r['dmD_se']), batch='stage-2 batch 1, R = 4000 (10 x 400), seed 20261013',
                              source_log=f'stage2/stage2_final.log:{ln} (Table 13): {txt}')))
un = json.load(open(os.path.join(PKG, 'stage3/unconstrained.json')))['records'][0]
assert un['id'] == 'U-L1657'
ln1, _ = log_line('stage3/stage3_unconstrained.log', 'Best evaluated point of the unbudgeted problem')
ln2, t2 = log_line('stage3/stage3_unconstrained.log', '  R=2000, seed 20261012: B = 423.96')
cands.append(dict(id='U-L1657', floor_lambda0=None, a=[float(v) for v in un['a']], chi=[float(v) for v in un['chi']],
                  p=[float(v)**2 for v in un['a']], source_params='code_8_3_stage2/stage2_pkg/stage3/unconstrained.json records[0]',
                  mc=dict(dmD=float(un['dmD_C']), paired_se=None, batch='none at R = 4000; R = 2000 (seed 20261012) value for reference only',
                          r2000_paired_se=float(un['dmD_se']), z='未定 (no R = 4000 verification batch; user decision 2026-10-20)',
                          source_log=f'stage3/stage3_unconstrained.log:{ln1} (best evaluated point, not an optimum); :{ln2}: {t2.strip()}')))
assert [c['id'] for c in cands] == ['NM-none-P5514', 'NM-0.3-P4312', 'AB-0.5', 'NM-0.7-UNIF', 'NM-none-P5450', 'P5014', 'U-L1657']

en = [stab(x, 'EN') for x in 'UDEN']; me = [stab(x, 'mE') for x in 'UDEN']
def ev_line(path, start):
    for i, l in enumerate(open(os.path.join(EV, path)), 1):
        if l.startswith(start): return i, l.strip()
    raise KeyError((path, start))
lr = ev_line('logs/run_all.log', 'EV B/U/n=24/8Delta EN '); lf = ev_line('logs/speedup_regression.log', 'REG B/U/n=24/8Delta EN_Phat fast')
B_U = float(lr[1].split()[-1]); assert B_U == float(lf[1].split()[4])
tN_raw = 10*max(v for _, _, v in en); tE_raw = 10*max(v for _, _, v in me)
cfg = dict(
    spec=dict(file='reports_CN/plan_recheck_bounds_CN.md', version='v4.2 (frozen; replaces v4.1 aab99e6)', commit='6863d7c'),
    predictions_commits=dict(first='2cfb3aa (v4.1 text)', amendment='pre-run spec correction to v4.2 (budget definition); see the note in predictions_recheck_bounds.md'),
    user_decisions_2026_10_20=['seven candidates = stage-2 finalists without UNIF + U-L1657; UNIF is the reference row',
                               'U-L1657: (rec - MC)/se column 未定 (no R = 4000 paired se)',
                               'budget relative to the uniform input, B := E[N](u_unif) (spec v4.2), not the absolute 316 of v4.1'],
    evaluator=dict(file='evaluator/evaluator_fast.py', version='B', n=24, margin='8 Delta', reference='UNIF, same evaluator'),
    candidates=cands, reference=dict(id='UNIF', a=[0.5]*4, chi=[0.0]*4, source='spec §A4; mc83.py:5'),
    tau_N=dict(value=ceil_1sig(tN_raw), raw=tN_raw, rule='10 x max over U, D, E, N of |E[N](12) - E[N](24)| (version B), rounded up to one significant figure',
               sources=[f'evaluator/logs/analysis.log:{i}: {t}' for i, t, _ in en]),
    tau_E=dict(value=ceil_1sig(tE_raw), raw=tE_raw, rule='10 x max over U, D, E, N of |m_E(12) - m_E(24)| (version B), rounded up to one significant figure',
               sources=[f'evaluator/logs/analysis.log:{i}: {t}' for i, t, _ in me]),
    budget_B=dict(value=B_U, definition='B := E[N](u_unif), evaluator_fast.py version B, n = 24, margin 8 Delta (the U row of this run; must equal this value bitwise)',
                  sources=[f'evaluator/logs/run_all.log:{lr[0]}: {lr[1]}', f'evaluator/logs/speedup_regression.log:{lf[0]}: {lf[1]}']),
    rules=dict(feasibility='budget rule 8.3.6(g) relative to the uniform input: E[N] <= B - tau_N: 按数值判据可行; E[N] >= B + tau_N: 按数值判据不可行; otherwise 边界未定; '
                           'the uniform input qualifies by definition (按定义合格). tau_N is a pre-frozen empirical grey zone, not an error bound',
               lambda0='candidates with a floor lambda0: m_E >= lambda0 + tau_E: 按数值判据满足; m_E <= lambda0 - tau_E: 按数值判据不满足; otherwise 未定; no floor: not applicable',
               mc_comparison='(rec - MC)/se with the paired Delta m_D se of the R = 4000 verification batch; no threshold, numbers only'))
json.dump(cfg, open(os.path.join(HERE, 'config_recheck.json'), 'w'), indent=1, ensure_ascii=False)
print(f"tau_N raw {tN_raw!r} -> {cfg['tau_N']['value']}; tau_E raw {tE_raw!r} -> {cfg['tau_E']['value']}; B = {B_U!r}")
for c in cands: print(c['id'], c['floor_lambda0'], c['source_params'], c['mc']['dmD'], c['mc']['paired_se'])
