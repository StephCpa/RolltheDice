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

事件 #4 登记（2026-10-10，外部评审发现，见 analysis/2026-10-10-external-review.md §1.1）：
evaluator.seen() 未截断 </a>。run_eval 批量 generate 时提前停止的行被补 pad（<|endoftext|>），
circle_canonicalize 因尾部内容返回 None → 已见解被计为"未见/新解"。影响全部 circle 运行的
test_seen（偏低）、test_unseen 与 n_unique_valid_novel（偏高）；test_validity、train_memorization、
n_unique_valid 不受影响。偏差随各臂超长输出比例变化，跨臂不抵消。原实现 predict() 已
split("</a>")[0]+"</a>"，故为移植特有问题。已修复（seen 截断＋run_eval 截断＋回归测试），
此后每次评价另存 preds_step*.jsonl。已有 final_model 需用 code/rescore.py 重评（含服务器上
修复前启动的 tl_hybrid_v2 / setc_m4_v2 / ntp_randperm_v2）。

事件 #5 登记（2026-10-10，评审第 5 轮代码核对发现）：code/review_analyses.py 的 memo 重评分
把预测整串（含本 seed 的 hash 前缀）与组员 target_text 整串比较；组员的 target_text 带各自的
hash 前缀，因而只可能与本 seed 的名义目标相等。memo_pi_final 与 memo_any_group_target 被迫退化为
"π*_final 不动点且被复现"的比例。核对：用已提交的 setc_m4_v2/churn.jsonl.gz 与固定种子重建的分组，
π*_final 在这 128 个训练探针上恰有 35 个不动点 = 35/128 = 0.2734，与三个 memo 读数完全一致。
因此 runs/review_analyses_20261010.json 的 A 部分与 stage0-report"增补 2"中"伪读数假说被否证 /
setc_v2 只复现 27% 训练对 / 坍缩≠记忆式过拟合"的裁定**作废**；现有数据（35 个不动点探针全部被复现）
反而与"setc_v2 完整记住了自选配对"相容，需用修正脚本或 rescore.py --pairing_file 在服务器上重评定案。
已修复（只比较答案部分；evaluator 新增 memo_own 与自检用例）。ntp_randperm_v2 的 memo_any_group_target
（0.164）同受影响，应以结构级 train_seen 替代。

事件 #7 登记（2026-10-11，分支作者核对修订 №1 §2 时发现）：修订 №1 锁定 tl_RN 的伙伴 R 为"非名义条件版"
（组内错排，与名义零重合）、种子 777；但实际运行的 `--fixed_pairing random` 抽的是**均匀随机双射**、随机流为
seed+2。按与 train.py 相同的调用重建：种子 42/43/44 下 R 与名义的**条目级重合率 = 0.243 / 0.243 / 0.242**
（规格要求 0），即约 24% 的 seed 的 TL 目标就是名义目标、没有冲突。影响：tl_RN/s42（增补 6 的预览读数）与
正在运行/排队的 ntp_2tgt、两臂种子 43/44 都是该变体，**不是**修订 №1 规格的运行。处理：
(1) 已有及在跑的运行改标 `tl_RN_unif` / `ntp_2tgt_unif`（探索性变体，保留——其 24% 无冲突比例恰与 setc_MN 的
π* 不动点占比 25–32% 相当，反而是与 MN 冲突率匹配的对照）；(2) train.py 新增 `--fixed_pairing derangement`
与 `--fixed_seed`，fixed_pairing.json 自动登记条目级/结构级重合率；`run_stage0b.sh conflict` 改为规格版
（derangement，种子 777），旧规格移入 `conflict_unif`；(3) 规格版需按修订 №1 补跑 3 种子后再判读分支表。
另：服务器 run_meta 的 git_commit 为 null（同步目录非 git 副本），train.py 现回退读取环境变量 P6_CODE_SHA。
