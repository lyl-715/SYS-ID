"""§8.5 stage 4: shared setup, exec'd by stage4_*.py.  Run from code_8_3_stage2/stage2_pkg/:
    exec(open('stage4_common.py').read())
Reuses, without modification: stage3_common.py, and through it stage2_common.py (mc83 setup, qual from mc85_post.py,
sod_batch + check_trigger, evaluate_crn, summarize_s2, pareto_front, eval_many, diff_se) and the Nelder-Mead block of
stage2_local.py (x_to_input, input_to_x, eval_s, pen, nm_run).
Adds bookkeeping only (rules in stage4/design.md): the loader of the stage-2/3 R=2000 pool, the keys of every stage-2/3
Nelder-Mead start, the R=2000 feasibility rule, the representative rule (design.md §3) and t-quantiles.
Outputs go to stage4/; stage2/ and stage3/ are only read."""
exec(open('stage3_common.py').read())
from scipy.stats import t as tdist
OUT = 'stage4'; os.makedirs(OUT, exist_ok=True)
S3 = 'stage3'                                         # read-only stage-3 outputs
FLOORS = TARGETS                                      # [('none', None), ('0.3', 0.3), ('0.5', 0.5), ('0.7', 0.7)]
SEED_SEL, SEED_CONF, SEED_SCEN = 20261018, 20261020, 20261021
T9 = float(tdist.ppf(0.975, 9))                       # exploratory 95% interval, 10 batches x 200 (Script 1)


def kkey(r): return key_of(r['p'], r['chi'])

def feasible(r, l0):
    """R=2000 feasibility for floor l0 (stages 2-3 rule): qual 'ok' and m_E^(C) >= l0 - 1e-9; l0=None: budget only"""
    return r['dec'] == 'ok' and (l0 is None or r['mE_C'] >= l0-1e-9)

def fails(r, l0):
    """which constraints r fails at R=2000 for floor l0 ('' if feasible)"""
    f = [] if r['dec'] == 'ok' else [f"budget ({r['dec']})"]
    if l0 is not None and r['mE_C'] < l0-1e-9: f.append(f"m_E<{l0}")
    return '+'.join(f)

def s23_pool():
    """every R=2000, seed-20261012 record of stages 2-3: stage2/pool_R2000.json (54), stage3/dense_R2000.json (56),
    stage3/unconstrained.json (10 U- endpoints); keys asserted unique."""
    P = json.load(open(f'{S2}/pool_R2000.json')); Dn = json.load(open(f'{S3}/dense_R2000.json')); U = json.load(open(f'{S3}/unconstrained.json'))
    for X in (P, Dn, U): assert X['R'] == R2 and X['seed'] == SEED2
    recs = P['records']+Dn['records']+U['records']
    ks = [kkey(r) for r in recs]; assert len(set(ks)) == len(ks), "duplicate keys in the stage-2/3 pool"
    return recs, P['ldC']

def used_starts():
    """key -> [(file, target, start id)] for the x0 of every Nelder-Mead task of stages 2-3 (design.md §1.1)"""
    u = {}
    for f in (f'{S2}/local_runs.json', f'{S3}/dense_runs.json', f'{S3}/unconstrained.json'):
        for t_ in json.load(open(f))['tasks']:
            a_, chi_ = x_to_input(t_[3]); u.setdefault(key_of(a_**2, chi_), []).append((f, t_[0], t_[2]))
    return u

def representatives(recs):
    """design.md §3: floors from the strictest (0.7) to none; each takes the feasible non-uniform candidate with the largest
    Delta m_D^(C) that is not already the representative of a stricter floor. Returns {floor: dict or None}."""
    reps = {}; chosen = set()
    for lab, l0 in reversed(FLOORS):
        feas = sorted([r for r in recs if feasible(r, l0)], key=lambda r: -r['dmD_C'])
        c = [r for r in feas if not is_uniform(r) and kkey(r) not in chosen]
        if c:
            reps[lab] = dict(id=c[0]['id'], floor_best=feas[0]['id'], replaced=c[0]['id'] != feas[0]['id']); chosen.add(kkey(c[0]))
        else: reps[lab] = None
    return {lab: reps[lab] for lab, _ in FLOORS}
