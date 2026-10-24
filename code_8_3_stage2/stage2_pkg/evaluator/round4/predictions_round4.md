# Predictions — round 4 ((k, r)-dependent threshold table, DP design, evaluation on the accepted grid)

Committed before any run of this round (before any code of this round is written, before any evaluator call, DP or
λ scan). The report `report_round4_CN.md` cites this file's commit hash and marks every prediction right / wrong /
partly right, wrong ones included.

The block below is copied verbatim from the frozen spec `reports_CN/plan_round4_CN.md` v3.1 (commit d15ac2e,
sha256 0f7b27a16580d04388c77db626f10e4fca5065474dcd3030b6f9d8f90560590e): §4 = spec lines 95–106, extracted with `sed -n '95,106p'`, not retyped.

---

# 4. 预测（跑前提交）

| 编号 | 预测 |
|---|---|
| V1 | 检查 1–6 全过（5 的利用率目标 $\le0.5$ 不计入通过） |
| V2 | 均匀输入、$S$ 目标：最终可行策略的 $S\in[520,600]$（常数阈值 466） |
| V3 | 均匀输入、加权目标：实算 $\Delta m_D\in[+0.5,+1.5]$ nat |
| V4 | 窗口表形状：$h_k$ 信息丰富的时段窗口窄，否则宽；参考值贴近窗边缘的节点窗口相对放宽 |
| V5 | P5514 上的实算 $\Delta m_D\in[+0.3,+1.0]$，小于均匀输入上的 |
| V6 | 加权目标策略的 $S$ 低于 $S$ 目标策略的 $S$，实算 $\Delta m_D$ 更高 |
| V7 | 一个 $\lambda$ 的 DP $\le30$ min；总计 $\le16$ h（预测；运行上限是 4 种情况各 40 次 DP，满额 160 次按 10–30 min 为 27–80 h，超过 16 h 只判预测错，不改算法或验收） |
| V8 | 最终可行策略的 $B-\mathbb E[N]\le0.5$（利用率目标） |
