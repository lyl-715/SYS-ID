# Predictions — round 7 (quadrature diagnosis, fine-grid regrade, SoD-referenced decomposition, candidate T4)

Committed before any run of this round: before any code of this round is run, before the Part 0 verification, before the
T4 restoration / DP re-run, before any evaluator call, forward pass or DP. The report `report_round7_CN.md` cites this
file's commit hash and marks every prediction right / wrong / partly right, wrong ones included.

The block below is copied verbatim from the frozen spec `reports_CN/plan_round7_CN.md` v3.1 (commit 9a75f5c,
sha256 cf29d395fe8c400b515063f5b1170b03d0114eedbdc9a80757c9dfb99ca20ed0): §4 = spec lines 62–71, extracted with
`sed -n '62,71p'`, not retyped.

---

# 4. 预测（跑前提交）

| 编号 | 预测 |
|---|---|
| G1 | 诊断：冻结点原 $\sigma/48$ 与 $\sigma/72$ 差 $>1$ 次，平移 $\sigma/48$ 与 $\sigma/72$ 差 $<0.1$ 次（支持分界取值解释） |
| G2 | 收敛：五候选 × 27 点，$\max|d_1|\le0.3$、$\max|d_2|\le0.05$；$d_1/d_2$ 中位在 $[8,20]$（二阶约 12.5），符号一致 |
| G3 | 参考网格：$m=0.10$ 在 P1 $+0.5$ 仍超 $B$（约 7 次）；T4 26 点全守；$\log\det$ 在两细网格间变化 $\le0.01$ nat |
| G4 | 分解（$m=0.20$）：$|$第二项$|\le0.1$ nat；T4 的 $c$ 在参考网格上 $\in[1.6,1.9]$ |
| G5 | 判定情形 3，选中 T4 |
| G6 | 工作量：T4 的 $n=24$ 27 点；$f=3,5$ 各 $6\times27=162$ 次前向（$\sigma/120$ 约 10 s）；参考网格名义曲线 841 次（约 2.3 h）；20 点局部匹配约 $20\times41$ 次（约 2.3 h），全扫回退另计；小网格检验；总 $\le6$ h，回退按次数增加 |
