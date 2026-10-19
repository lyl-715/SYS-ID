# Evaluator acceptance bundle (spec v4.1) — unzip at the repo root, on top of dde7201

reports_CN/plan_evaluator_CN.md            frozen spec v4.1 (the contract); .pdf = same, typeset
reports_CN/sim_methods_8_5_CN.md           §8.5 methods note (context for the candidates)
reports_CN/section_8_3_single_CN.md        §8.3 single-scenario report; line 527 gives the old log det 197.197 with batch se 0.003 (R=10000, seed 20261006)
code_8_3_stage2/stage2_pkg/density_recursion.py   v0 prototype (38 lines). It exec()s mc83.py and loads
                                            wk_d20_10000.npy from its own directory — both already live here.
code_8_3_stage2/stage2_pkg/amp_nm_P5514.json      candidate D parameters (source citation)
review_8_5_stage2.zip                      format sample for the Step 6 review package (REVIEW_GUIDE.md inside)
handoff_ref/mc83.py, handoff_ref/README.md local copies from the handoff bundle, for diffing only —
                                            NOT a second mc83.py to run; the one under stage2_pkg/ is authoritative
                                            once the diff is clean.
