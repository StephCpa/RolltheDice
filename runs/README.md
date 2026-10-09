# runs 目录说明

只新增、不覆盖。每个运行目录包含：`run_meta.json`（臂、数据、超参、环境、启动时间）、`train.log`、`scores.jsonl`（逐次评价：train_memorization / test_seen / test_validity / test_unseen / n_unique_valid / n_unique_valid_novel）、`throughput.json`（每步秒数）、`final_model/`（末模型+tokenizer）、setc 臂另有 `churn.jsonl`（每次组访问的 π* 与代价）。

## 20261009-circle-10k —— 首轮 10k 步（已中止的中间版本）

五臂各 10k 步。结果：ntp_noseed 在 10k 步全指标为 0；抽样检查显示输出格式完全正确、贪心锁定同一"双环"（度数全 2 但不连通）结构——属训练量不足（原配置 40k 步、warmup 2000，10k 仅 1/4 训练量）而非实现错误。据此中止其余臂的 10k 版本，改跑完整 40k（见下）。10k 目录保留作轨迹与吞吐记录。

## 20261009-circle-40k —— 阶段 0 正式运行（进行中）

五臂 × **40k 步**（与原仓库 train.sh 相同）、每 5000 步评价（256 探针，train/test 各 128，greedy，与原 predict 机制一致）、其余超参与 10k 版相同。预计总时长约 8 小时（RTX 2080 Ti）。

五臂 × 10k 步、每 5000 步评价（256 探针：train/test 各 128，greedy）、bs 64、lr 1e-4、warmup 2000、fp16、seed 42、gpt2＋16 新 token（词表 50273）：

| 臂 | 数据 | 说明 |
|---|---|---|
| ntp_noseed | circle.10.9.0.10000 | 无 seed 基线 |
| ntp_fixed | circle.10.9.10.10000 | 原论文固定配对 |
| ntp_randperm | 同上 | G1 对照：持久 seed 集＋每轮随机双射 |
| tl_hybrid | circle_hybrid…（1:3 展开） | 原 teacherless 混合 |
| setc_m4 | circle.10.9.10.10000 | WP2 集合耦合（m=4，Hungarian） |

**环境差异登记（相对原仓库 train.sh 语义）**：
1. 硬件 RTX 2080 Ti 11GB（原配置 A6000 48GB）；吞吐数值不可直接对比 A6000。
2. 干净重实现（vendored simpletransformers 依赖 2021 版 transformers，无 Python 3.12 轮子）；语义逐条对照来源代码行（见 code/train.py 文档串）。
3. 动态 padding 到批内最长（≤128），与固定 pad 128 在因果 LM＋-100 掩码下数学等价（显存优化）。
4. 生成停在 `</a>`（原实现用默认 eos，评测按 `</a>` 切分；对格式良好的输出等价）。
5. 数据由 datagen.py 以固定种子 20261009 重生成（原 notebook 未固定种子）。
6. churn/评分中的"组"为随机分组（m=4，2500 组），固定配对臂的"配对"即数据本身的 hash↔target。

smoke/ 为 60–150 步烟雾测试；churn_synth* 为诊断工具的合成验证。全部材料 development 状态。

## 迁移登记（2026-10-10）

按用户指示停止本地实验（机器负载过高），迁移至服务器 gpu-gfkd（gfkd-NF5468M6，
8× RTX A6000 49GB，torch 2.4.0+cu121 / transformers 4.47.1 / scipy 1.14.1，
HF 走 hf-mirror）。本地终止情况：tl_hybrid_v2 于 step ~17000/40000 终止（剩 ~54 分钟
>30 分钟阈值），部分记录保留（见其 PARTIAL.md）；重跑链 watcher 一并停止；本地 GPU 清空。

服务器运行（~/P6-Roll_the_dice/runs/server-circle-40k/，GPU 4 双流并行，2026-10-10 00:47 启动）：
- 流 A：tl_hybrid_v2（40k，修 bug 后的完整重跑）→ setc_m4_v2（40k，含 NTP 配对修复＋churn＋4×4 代价矩阵转储）
- 流 B：ntp_randperm_v2（40k，真正的 G1 随机双射对照——本地同名臂因事件 #3 失效）

事件 #3 登记（2026-10-10，由外部评审对 memorization 数字的追问触发自查发现）：
train.py group_step NTP 前向误用 groups_ex[g][j] 整个元素（前缀+答案均取 j），
导致 ntp_randperm 首轮从未真正重配（等价组批版 ntp_fixed，该臂标记 INVALID）、
setc_m4 首轮 NTP 分量（1/4 权重）训练名义配对（标记 DEVIATION，MTP 匹配分支正确）。
已修复（前缀取 i、答案取 π(i)）并烟雾验证；本地 setc_m4/ntp_fixed/ntp_noseed 不受影响。
