"""Part 0 memory probe (report §3/§4 item 1): peak RSS (VmHWM, kB) of one sigma/24 + sigma/120 evaluation of the U table T1 without
eval13.Stats, and of one sigma/24 evaluation with eval13.Stats followed by Stats.result(). Needs the T1 table of results/gate_U_design in
$ROUND13_WORK. Usage: R13_INPUT=U python3 mem_probe.py plain|stats"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import eval13 as e13
def hwm(): return int([l for l in open('/proc/self/status') if l.startswith('VmHWM')][0].split()[1])
st = json.load(open('results/gate_U_design/scan_edge_j1.json')); fin=[r for r in st['runs'] if r['run']==st['final_run']][0]
tab = e13.Table11(np.load(fin['table'], mmap_mode='r'), e13.edge_set(), 'T1')
print('after import', hwm())
if sys.argv[1] == 'stats':
    S = e13.Stats(np.ones(e13.K)); rec, J, fw = e13.evaluate(tab, stats=S); print('after forward with Stats', hwm(), 'nodes', sum(x.size for x in S.ev)); r = S.result(); print('after Stats.result', hwm())
else:
    rec, J, fw = e13.evaluate(tab); print('after forward', hwm())
    rf, Jf, ff = e13.evaluate_refined(tab, 5); print('after sigma/120', hwm())
