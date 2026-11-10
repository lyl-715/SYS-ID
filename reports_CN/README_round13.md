# Round 13 — does a designed trigger decouple communication from the input?

## Purpose

Paper 1 designed the input under constant-threshold send-on-delta and gained 0.70 nat. Rounds 4–11 designed the trigger for a fixed uniform input and gained 10.4 nat, within 0.07 nat of the best table of its class. This round asks whether the two designs interact:

> With constant SoD the number of transmissions follows the motion of the output, so the input is constrained by the budget. With a designed window table the number of transmissions is set by a price. **Hypothesis (separation): once the table is designed for the input, the information equals the classical full-data information of that input plus an input-independent constant** — so input design reduces to classical input design.

The round is a two-day feasibility test for a possible thesis topic (joint input–trigger design). It tests the hypothesis on a handful of inputs and two criteria; it proves nothing.

## Three prompts, in this order

| file | what it does | blocking? |
|---|---|---|
| `prompt_round13_part0_infrastructure_EN.md` | per-input copies of the round-11 evaluator and DP (the originals are bound to input `U` at import), helper functions, five regression gates | yes — nothing else runs until every gate passes |
| `prompt_round13_partA_input_invariance_EN.md` | six inputs, 9×9 D-criterion: full-data information, matched constant SoD, designed table with certificate; verdict H1–H3 | — |
| `prompt_round13_partB_parametric_EN.md` | criterion = information about (gain, time constant); classical optimal input `PAR`; four inputs; verdict H4–H5 and the gains G1–G5 | needs Part A's stored matrices |

Each part: commit its predictions file first, then run, then write its report (CN + EN copy) and commit. Reports follow the earlier rounds' rules: conclusion first; numbers with units and the grid they were computed on; quoted log lines for every check; no causal language; every unforced choice as `DECISION 13-x`.

## Facts about the code that the prompts rely on (checked at `round11-edge-actions` = `1b88f01`)

- `round11/eval11.py` line 43 and 56: `NAME = 'U'`; `XK, HK, Z, DZ, KA, KB = ew.design_grid(NAME)` at module level; `round11/dp11.py` copies these globals. One process serves one input.
- `round11/eval11.py` line 329: `er.cell_index(zf)` uses round 7's default coarse origin, which is input `U`'s.
- `density_recursion_v1.input_arrays(name)` reads `CFG['inputs'][name]`; `round4/evaluator_window.py` lines 56–58 already add `P5514` in memory.
- `Scan11.run` stores `dual = g1 + dsum + lam*(B − 1)` per run and `min_dual` per scan; the budget constant is `ew.B_BUDGET = 316.8921168493721`.
- `eval11._step_full` stops if any node's window edge leaves `[−X_PAD, G−1+X_PAD]` (`X_PAD = 960`), also for nodes without mass.
- Reference values for input `U` are in `round11/results/scan_edge_j0.json`, `scan_edge_j1.json`, `round11.json → tables.edge_j0 / edge_j1`, `table_edge_A0.npz`, `table_edge_iter.npz`.

## Numbers computed independently before the prompts were written (for identification, not as targets)

| input | Σa² | output range (σ) | largest step (σ) | G | 9×9 I_full | I^par_full |
|---|---|---|---|---|---|---|
| U | 1 | 1144.4 | 3.99 | 35146 | 219.0085 | 48.0196 |
| P5514 | 1 | 1136.2 | 3.18 | 34951 | 219.0080 | 48.0019 |
| SCH | 1 | 1121.4 | 3.03 | 34596 | 219.0085 | 47.9450 |
| LO | 1 | 1347.2 | 4.28 | 40015 | 218.0343 | 48.4359 |
| HI | 1 | 869.1 | 3.44 | 28540 | 218.0343 | 47.2803 |
| N | 1 | 1482.5 | 4.04 | 43262 | 218.4169 | 48.5188 |
| PAR | 1 | 1743.2 | 4.10 | 49518 | 202.4069 | 49.2509 |

## How the result will be used

- H1–H3 and H4–H5 hold → separation is the working hypothesis of the thesis topic; next step is to prove it and to exploit it (classical input design + trigger design).
- They fail clearly → input and trigger interact; joint design is a harder problem and the size of the interaction is the first result.
- Mixed or within the resolution of the certificate → the input matters little here; look elsewhere.

## Runtime

Part 0 about 1 h (the `U` gate is two full scans). Part A about 50 min per input, six inputs, four in parallel. Part B about 45 min per input, four inputs, four in parallel. Each process holds about 1.5 GB (more for the larger grids of `N` and `PAR`).
