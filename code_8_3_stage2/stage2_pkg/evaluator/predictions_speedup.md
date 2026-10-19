# Predictions — window speed-up of the accepted evaluator (version B only)

Committed before the kernel zero test (spec §2) and before any run of the windowed path. The speed-up report cites this
file's commit hash and marks every prediction right / wrong / partly right, wrong ones included (a wrong node-count
prediction is reported, not a failure).

The block below is copied verbatim from the frozen spec `reports_CN/plan_speedup_CN.md` v3.1 (commit d4db8a6),
§6 = spec lines 86–94, extracted with `sed -n`, not retyped. (`predictions.md` in this directory belongs to the
acceptance run and is unchanged.)

---

# 6. 预测（在 §2 的核零值测试之前单独提交一次 commit，报告引用其 hash）

| 编号 | 预测 |
|---|---|
| S1 | 12 次回归全部通过，最大相对差 $<10^{-13}$（只有求和顺序差异） |
| S2 | 区间内非零节点数（初始化与后续更新分开报）：初始 $m_2$ 的支撑宽 $<78\sigma$（$n=24$ 时 $<1872$ 个节点）；后续每步 $\le118\sigma$（$\le2832$ 个）。这两个都是上界；下限不预测（下溢和归一化可能进一步减少非零节点）。节点数只作预测对照，不是回归验收条件 |
| S3 | B/$n=24$ 单次耗时中位数 $\le2.5$ s（numba 循环从 13.2 s 降到约 1 s，`cfun` 的 1.1 s 不变成为主项） |
| S4 | 耗时随 $n$ 线性：$n=12$ 约 1.3 s，$n=6$ 约 0.7 s |
| S5 | 尾部检查差异与验收版同量级（$\le10^{-11}$） |
