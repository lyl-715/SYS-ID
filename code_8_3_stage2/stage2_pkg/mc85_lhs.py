"""§8.5 stage 2a: 3000 joint-space candidates, p ~ Dirichlet(1,1,1,1), chi ~ LHS-stratified Uniform[0,2pi)^4,
R=200 on the stage-1 noise draws (seed 20261010). Run in parts: python3 mc85_lhs.py k.
Part convention (the one the shipped mc85_lhs_part*.json follow): part 0 holds indices 0-499 (500 candidates);
parts k = 2..11 hold 250 each, indices 250k .. 250k+249 (500-2999); there is no part 1 (its range 250-499 lies in part 0)."""
import numpy as np, json, sys, time
exec(open('mc85_scan.py').read().split('part = int(sys.argv[1])')[0])   # setup, evaluate, summarize, reference, Z
part = int(sys.argv[1]); n_tot = 3000
assert part == 0 or 2 <= part <= 11, "part convention: 0 (indices 0-499) or 2..11 (250 each); there is no part 1"
idx = range(0, 500) if part == 0 else range(250*part, 250*(part+1))
rs = np.random.default_rng(20261011)
P = rs.dirichlet(np.ones(4), size=n_tot)
# LHS per phase coordinate: stratify [0,2pi) into n_tot cells, one draw per cell, random permutation per coordinate
U = np.zeros((n_tot, 4))
for m in range(4):
    U[:, m] = (rs.permutation(n_tot)+rs.random(n_tot))/n_tot*2*np.pi
cands = [(np.sqrt(P[i]), U[i]) for i in idx]
t0 = time.time(); out = [summarize(a, chi, evaluate(a, chi), 'lhs') for a, chi in cands]
print(f"part {part}: {len(out)} candidates, {time.time()-t0:.0f}s")
json.dump(out, open(f'mc85_lhs_part{part}.json', 'w'))
