# Circle exposure-control 开发重评分记录

日期：2026-10-17  
状态：`development`，不是 Sibling 确认结果，也不是新的预注册终点。

## 目的

在等待正式 Sibling 运行期间，使用已完成的 Circle development checkpoint，按
明确的 checkpoint 身份和 `task=circle` 重评分，核对 `2hash` 与固定配对、batch
对照之间的曝光量差异。该记录不改变既有选择规则，不生成新的训练模型，也不把
单 seed 结果写成泛化结论。

## 冻结的执行身份

| 臂 | seed | checkpoint | data / probe | 评价器 |
|---|---:|---:|---|---|
| `ntp_fixed` | 42 | `step25000` | `circle.10.9.10.10000` / `heldout.json` | `p6-evaluator-v2`, `task=circle` |
| `ntp_fixed_b128` | 42 | `step10000` | `circle.10.9.10.10000` / `heldout.json` | `p6-evaluator-v2`, `task=circle` |
| `ntp_fixed_2hash` | 42 | `step35000` | `circle.10.9.10.10000.dup2.20261017` / `heldout.json` | `p6-evaluator-v2`, `task=circle` |

checkpoint 由 development-only selection 记录产生；正式 heldout 评价只读取已选
checkpoint，没有依据 heldout 分数回改选择。

## 单 seed 结果

| 臂 | creativity | creativity_raw | test_validity | test_seen | unique novel valid |
|---|---:|---:|---:|---:|---:|
| `ntp_fixed` | 0.178223 | 0.178711 | 0.227051 | 0.048340 | 365 |
| `ntp_fixed_b128` | 0.159668 | 0.160645 | 0.209473 | 0.048828 | 327 |
| `ntp_fixed_2hash` | 0.219727 | 0.220215 | 0.428223 | 0.207520 | 450 |

相对差值（仅作开发记录）：

- `2hash − fixed` = `+0.041504` creativity；
- `2hash − b128` = `+0.060059` creativity；
- `b128 − fixed` = `−0.018555` creativity。

三个结果均显示 `train_memorization=1.0`（b128 的选定早期 checkpoint 除外，
其 `train_memorization=0.773438`），因此不能据此解释为“降低记忆”。这些结果
只能支持后续把 `2hash` 作为 Sibling 独立确认候选，并不能替代三种训练 seed 的
fresh heldout 评价。

## 证据边界

1. 这是 Circle 的单 seed、development checkpoint 选择后的重评分；不计入 Sibling
   确认的 seed 数，也不改变既有论文主终点。
2. `2hash` 的候选解释仍是固定多 hash 入口；`repeat_rows_k2` 曝光控制尚未执行。
3. 任何跨任务或规模结论必须等待 Sibling 三臂确认和后续预先冻结的比较。

原始远程输出位置：

`/home/wlwuser/P6-Roll_the_dice/runs/icml2027-dev-20261017/exposure-gpt2/`

