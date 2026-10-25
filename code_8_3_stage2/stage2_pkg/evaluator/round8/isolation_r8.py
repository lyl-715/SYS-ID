"""Round 8, Part A2: true-value isolation of the design module by code dependency (spec reports_CN/plan_round8_CN.md v3.1,
frozen; §1.3, §3 item 3).

1. Static import graph (import_graph / static_check): starting from design_r8.py, every `import` / `from ... import`
   statement (module level and inside functions) is followed through the round-8 directory; the repo-local modules reached
   must be exactly the allowlist ALLOWED (declared, the generated verbatim mirrors r8k_*, design_r8 itself) and their
   sources must not use any name in FORBIDDEN_NAMES (theta_0 holders and every frozen module that executes the mc83 setup
   block) nor exec / eval / compile / __import__ / importlib / open. Third-party imports are listed (numpy, scipy, numba and
   the standard library only).
2. Dynamic import check (dynamic_check): a fresh interpreter (python3 -I) imports design_r8 and lists sys.modules: no
   forbidden module may be loaded and no module from the repository outside ALLOWED.
3. Runtime sentinel (guard): while a design runs in the pipeline, every theta_0 holder in the loaded modules (truth_r8.THETA0,
   X0, its mc83 namespace; posthoc_r8.THETA0, X0; density_recursion_v1.x, G, Gj, output, input_arrays; robustness.THETA0,
   INP; evaluator_refine.I24) is replaced by a Sentinel object that raises SentinelAccess (a RuntimeError, deliberately not an
   AttributeError) on any attribute access, call, indexing, iteration, numeric conversion, comparison or arithmetic, and
   records the access in ACCESS_LOG; the originals are restored afterwards and the caller asserts ACCESS_LOG is empty.
   SentinelAccess is a correctness failure: it is never caught by the design-failure fallback.
"""
import os, sys, ast, json, subprocess
from contextlib import contextmanager

HERE = os.path.dirname(os.path.abspath(__file__))
ALLOWED = ('design_r8', 'declared', 'r8k_dr', 'r8k_ef', 'r8k_ew', 'r8k_td')
FORBIDDEN_MODULES = ('truth_r8', 'posthoc_r8', 'seq_pipeline', 'stats_r8', 'checks_r8', 'mc83', 'mc84', 'mc836', 'mc837',
                     'mc85_scan', 'mc85_post', 'stage2_common', 'density_recursion', 'density_recursion_v1', 'evaluator_fast',
                     'evaluator_window', 'trigger_dp', 'robustness', 'evaluator_refine', 'two_stage', 'margin_design')
FORBIDDEN_NAMES = set(FORBIDDEN_MODULES) | {'THETA0', 'theta0', 'theta_0', 'Gj', 'X0', 'xbar', 'I24', 'INP', 'H_MC83'}
FORBIDDEN_CALLS = {'exec', 'eval', 'compile', '__import__', 'open', 'importlib'}
STDLIB_OK = {'os', 'sys', 'time', 'math', 'json'}
THIRD_OK = {'numpy', 'scipy', 'numba'}


# --- static ----------------------------------------------------------------------------------------------------------------
def _imports(path):
    tree = ast.parse(open(path).read()); mods = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import): mods += [a.name.split('.')[0] for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0: mods.append(n.module.split('.')[0])
    return tree, sorted(set(mods))

def import_graph(start='design_r8'):
    """repo-local modules reachable from `start` through import statements (round-8 directory and the frozen trees)."""
    seen, third, edges = set(), set(), []
    todo = [start]
    while todo:
        m = todo.pop()
        if m in seen: continue
        seen.add(m)
        _, mods = _imports(os.path.join(HERE, m+'.py'))
        for x in mods:
            edges.append((m, x))
            if os.path.exists(os.path.join(HERE, x+'.py')): todo.append(x)
            elif x in FORBIDDEN_MODULES: seen.add(x)                  # a frozen theta_0 holder outside round8/: recorded as reached
            else: third.add(x)
    return sorted(seen), sorted(third), edges

def static_check(start='design_r8', allowed=ALLOWED):
    local, third, edges = import_graph(start)
    bad_mod = [m for m in local if m not in allowed]
    bad_third = [m for m in third if m not in THIRD_OK | STDLIB_OK]
    bad_names = {}
    for m in local:
        p = os.path.join(HERE, m+'.py')
        if not os.path.exists(p): continue
        tree = ast.parse(open(p).read()); hits = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and (n.id in FORBIDDEN_NAMES or n.id in FORBIDDEN_CALLS): hits.add(n.id)
            elif isinstance(n, ast.Attribute) and n.attr in FORBIDDEN_NAMES: hits.add('.'+n.attr)
            elif isinstance(n, ast.alias) and n.name.split('.')[0] in FORBIDDEN_NAMES | {'importlib'}: hits.add('import '+n.name)
        if hits: bad_names[m] = sorted(hits)
    ok = not bad_mod and not bad_third and not bad_names
    return dict(start=start, local=local, third_party=third, edges=edges, outside_allowlist=bad_mod, bad_third_party=bad_third,
                forbidden_names=bad_names, passed=ok)

def dynamic_check(start='design_r8'):
    code = (f"import sys, json; sys.path.insert(0, {HERE!r}); import {start}; "
            "print(json.dumps(sorted([(n, getattr(m, '__file__', None)) for n, m in list(sys.modules.items())])))")
    out = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, cwd='/', timeout=600)
    if out.returncode != 0: return dict(passed=False, error=out.stderr[-2000:])
    mods = json.loads(out.stdout.strip().split('\n')[-1])
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    loaded_forbidden = [n for n, f in mods if n.split('.')[0] in FORBIDDEN_MODULES]
    repo_local = sorted({n for n, f in mods if f and os.path.abspath(f).startswith(repo+os.sep)})
    outside = [n for n in repo_local if n not in ALLOWED]
    return dict(passed=not loaded_forbidden and not outside, repo_local_loaded=repo_local, forbidden_loaded=loaded_forbidden,
                outside_allowlist=outside, n_modules=len(mods))


# --- runtime sentinel --------------------------------------------------------------------------------------------------------
ACCESS_LOG = []

class SentinelAccess(RuntimeError):
    """a theta_0 holder was touched while a design was running (correctness failure; never a design failure)."""

def _trip(self, what):
    name = object.__getattribute__(self, '_name')
    ACCESS_LOG.append((name, what))
    raise SentinelAccess(f"theta_0 holder {name} accessed during design ({what})")

class Sentinel:
    def __init__(self, name): object.__setattr__(self, '_name', name)
    def __getattribute__(self, a): _trip(self, f'attribute {a}')
    def __setattr__(self, a, v): _trip(self, f'set attribute {a}')
    def __call__(self, *a, **k): _trip(self, 'call')
    def __getitem__(self, i): _trip(self, 'getitem')
    def __setitem__(self, i, v): _trip(self, 'setitem')
    def __len__(self): _trip(self, 'len')
    def __iter__(self): _trip(self, 'iter')
    def __contains__(self, x): _trip(self, 'contains')
    def __array__(self, *a, **k): _trip(self, '__array__')
    def __bool__(self): _trip(self, 'bool')
    def __float__(self): _trip(self, 'float')
    def __int__(self): _trip(self, 'int')
    def __index__(self): _trip(self, 'index')
    def __repr__(self): _trip(self, 'repr')
    def __str__(self): _trip(self, 'str')
    def __format__(self, spec): _trip(self, 'format')
    def __hash__(self): _trip(self, 'hash')
    def __eq__(self, o): _trip(self, 'eq')
    def __ne__(self, o): _trip(self, 'ne')
    def __lt__(self, o): _trip(self, 'lt')
    def __le__(self, o): _trip(self, 'le')
    def __gt__(self, o): _trip(self, 'gt')
    def __ge__(self, o): _trip(self, 'ge')
    def __neg__(self): _trip(self, 'neg')
    def __pos__(self): _trip(self, 'pos')
    def __abs__(self): _trip(self, 'abs')
for _op in ('add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'pow', 'matmul'):
    setattr(Sentinel, f'__{_op}__', (lambda op: lambda self, o: _trip(self, op))(_op))
    setattr(Sentinel, f'__r{_op}__', (lambda op: lambda self, o: _trip(self, 'r'+op))(_op))

HOLDERS = {'truth_r8': ('THETA0', 'X0', '_ns', 'H_MC83'), 'posthoc_r8': ('THETA0', 'X0'),
           'density_recursion_v1': ('x', 'G', 'Gj', 'output', 'input_arrays'), 'robustness': ('THETA0', 'INP'),
           'evaluator_refine': ('I24',)}

@contextmanager
def guard():
    """replace every theta_0 holder of the loaded modules by a Sentinel for the duration of the block; restore afterwards."""
    saved = []
    for mn, attrs in HOLDERS.items():
        m = sys.modules.get(mn)
        if m is None: continue
        for a in attrs:
            if hasattr(m, a):
                saved.append((m, a, getattr(m, a))); setattr(m, a, Sentinel(f'{mn}.{a}'))
    try:
        yield [f'{m.__name__}.{a}' for m, a, _ in saved]
    finally:
        for m, a, v in saved: setattr(m, a, v)
