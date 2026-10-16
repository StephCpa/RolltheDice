# P6 阶段 0b 执行方案（评审第 1–5 轮收敛版，含预注册）

状态：**方案稿，未执行**。依据：`runs/stage0-report.md`（含增补 1–3）、`analysis/2026-10-10-external-review.md`、
五轮评审意见。预算：阶段 0 余额（约 48 A6000·时）。代码：`code/run_stage0b.sh` 一键分阶段执行；
全部机制已有回归测试（`python code/test_pipeline.py`，27 项，CPU 离线约 12 秒；关键项经变异检验）；启动器已在 CPU 上以微型模型、64 例数据、步数缩放后**端到端演练全部阶段**（A→B→lock→intervene→C→replay→diag→heldout→report）。重构后的训练器对所有既有预设与旧版**逐位一致**（权重、churn、代价矩阵）；完整状态续训与直跑逐位一致。

## 0. 主张登记（启动前状态）

| 主张 | 状态 | 依据 |
|---|---|---|
| 集合耦合已有初步生成优势 | **撤回** | 同环境下 setc_v2 终点 creativity 0.023 vs ntp_fixed_v2 0.180；旧领先来自偏差版 |
| 最优配对"摆脱永久配对" | **撤回** | 两版均在 ~12–14k 完全固化 |
| G1："非永久性无益" | **作废**（事件 #3） | 首轮 randperm 从未重配 |
| randperm 早期"新颖峰值最高"（0.258） | **更正** | 那是 novel-valid（含重复）；主指标 creativity 下峰值 0.070（20k，9 个唯一结构）< fixed 0.227——**不是强基线**，但仍纳入检查点选择协议 |
| "增补 2"：setc_v2 只复现 27% 训练对、非记忆式过拟合 | **作废**（事件 #5） | 整串比较 bug；0.2734 = π*_final 在探针上的不动点数 35/128，且 35/35 均被复现 |
| C01"同代价 3.89×" | **撤回** | 改为：同目标复用（C01a）另测；3.89× 是跨目标路线比（C01b） |
| 三阶段固化轨迹 | **保留（描述性）** | v1/v2 复现；承诺时间中位 11.5k（v2）/10.8k（v1）；v2 冻结排列循环型≈均匀 S₄，v1 偏向恒等（不动点 0.316，名义 NTP 的足迹） |
| tl_hybrid_v2 validity 0 / memo 1.0 | **非移植 bug（已核）**；与论文数字的关系未定 | 数据（`circle_hybrid.ipynb`）、标签对齐（`preprocess_data_bart`：target_labels、pad/前缀置 -100）、超参（teacherless/train.sh 与 ntp 相同、数据为 hash 版）逐项一致 |

## 1. 收敛后的核心问题

固化之后，setc_v2 与 ntp_fixed **都在训练一个固定双射**，却一稳一崩。差别只剩三项：(a) 哪个双射；(b) 前 ~14k 步的动态；
(c) setc 流水线本身——其中最大的一项是 **3× 权重的 teacherless（TL）损失**。现有四个数据点：

| TL 目标 \ NTP 目标 | 名义 N | 匹配 M |
|---|---|---|
| **名义 N** | tl_hybrid_v2（DataLoader 版）：memo 1.0、validity **0** | 未测（反向组合） |
| **匹配 M** | 偏差版 setc v1（本地）：**稳定**（0.250 / 0.195） | setc_v2：记住 π* 后**坍缩** |

**首要待检假说 H-TL**：TL 槽位只看得到 hash，TL 损失对 hash 而言就是查表目标；两分支目标一致时查表可被完全拟合并压垮
未见 hash 上的结构生成，两分支目标不一致（v1：约 68% 的 seed 同时有两个目标）时不能。H-TL 与"名义锚定正则"、"多目标训练"
是同一现象的不同说法，区分它们正是下面 2×2 的任务；**全部在同一组级流水线、同算力、同环境内比较**。

## 2. 运行矩阵

所有训练臂：`--eval_every 5000 --ckpt_every 5000 --save_final`，A6000，同一 git 提交（run_meta 记录 `git_commit`），
TL/setc 臂另存 churn 与代价矩阵。标签即 `runs/stage0b/<标签>/s<seed>/`。

### 阶段 A：不训练（约 1 GPU·时）—— `run_stage0b.sh A`
- A1 已有末模型重评：memo_own（setc_v2 用 `pistar_final.json`，本仓库已从 churn 导出并核对为双射）、train_seen、creativity；
  无 seed 臂温度采样 T∈{0.5,1.0}（同生成数对照）。
- A2 同口径间隔基线：`diagnose.py --mode gap`（AR 代价、自身配对 vs 其余 23 个排列、按 token 与 dropout 噪声归一）——
  ntp_fixed_v2（名义）vs setc_v2（π*）。取代 ntp_fixed 上的 TL 间隔（分布外读数）。
- A3 C01：`throughput.py` 于 setc_v2 末模型，m∈{4,8,16}：C01a 同目标复用（先核一致再计时）、C01b 跨目标决策一致率与遗憾。
- A4 tl_hybrid_v2 保真度：teacherless 并行读取 + 温度采样读取（论文提示 teacherless 需采样读出）。

### 阶段 B：流水线自检 + TL×NTP 2×2（种子 42，约 12 GPU·时）—— `run_stage0b.sh B`

| 标签 | 命令要点 | TL / NTP 目标 | 回答 |
|---|---|---|---|
| `ntp_fixed` | `--arm ntp_fixed` | — / N | 原论文设定（带检查点重跑，供选择协议） |
| `ntp_fixed_grp` | `--arm ntp_fixed_grouped` | — / N（组批） | 流水线自检：须 ≈ ntp_fixed（即首轮 INVALID randperm 的配置） |
| `tl_NN` | `--arm tl_fixed_m4` | N / N | H-TL 预测：像 tl_hybrid 一样失去未见 hash 有效率 |
| `setc_MM` | `--arm setc_m4 --state_at 10000,14000,20000` | M / M | setc_v2 带检查点重跑；分支点 |
| `setc_MN` | `--arm setc_m4 --ntp_lambda 1` | M / N | **偏差版 v1 的精确重建**（同环境）；H-TL 预测稳定 |
| `setc_NM` | `--arm setc_m4 --tl_target nominal` | N / M | 反向组合；H-TL 预测稳定（两分支目标不一致） |

附加（便宜，`run_stage0b.sh lock` / `intervene`）：
- **冻结-锁定等价检验**：从 `setc_MM/s42/state_step14000` 续训到 16000 两支——在线 vs `--lock_at 14000`；
  `diagnose.py --mode compare`。锁定不改变任何随机数消耗（TL 前向本为 TL 损失所需），若 14k 时已全固化则两支**逐位相同**；
  有差异先查实现/随机流，不得解读为"在线匹配的隐性收益"。
- **晚期干预**：从 `state_step20000`（坍缩前）以 `--ntp_lambda 1` 续训到 40k。负结果只说明"晚加无效"，不否定早期作用。

**阶段 B 后的预注册决策**（单种子，只决定阶段 C 跑什么，不形成主张）：

| 观察 | 决策 |
|---|---|
| `ntp_fixed_grp` 与 `ntp_fixed` 选中检查点留出 creativity 差 > 0.05 | 流水线问题（c）：停，排查后再继续 |
| `tl_NN` 坍缩、`setc_MN` 与 `setc_NM` 稳定 | H-TL 获支持：阶段 C 以 "两分支目标一致/不一致" 为主变量；匹配本身降为次要 |
| `setc_MN` 同环境也坍缩 | v1 的稳定不可复现：撤回锚定/多目标假说 |
| `tl_NN` 稳定、`setc_MM` 坍缩 | 问题在自选配对或早期动态：阶段 C 优先 π* 回放 |

### 阶段 C：确认（种子 43、44，并入 B 的 42；约 29 GPU·时）—— `run_stage0b.sh C` / `replay`
- C1：`ntp_fixed`、`setc_MN`、`setc_MM`、`tl_NN` × 3 种子；`setc_M05`（`--ntp_lambda 0.5`）× 3 种子。
  λ 只取 {0, 0.5, 1}，不做细网格（评审：单种子 5 点网格取峰必然"找到"赢家）。
- C2 π* 回放（种子对齐：各种子用自己 `setc_MM` 运行导出的 π*）：
  `tl_NN --pairing_file π*`（全目标回放，对照 `tl_NN`）与 `ntp_fixed --pairing_file π*`（仅 NTP 回放，对照 `ntp_fixed`）。
  回放用的配对是训练后才得到的，只作机制实验，不计为"免费"的算法收益。
- 可选 C3（若 B/C 有正结果）：sibling 上 `ntp_fixed` / `setc_MN` / `tl_NN` 各 1 种子，检查是否 circle 特有。

## 3. 评价协议（预注册，启动前由项目负责人签字确认 margin）
1. **主指标** creativity = 去重（canonical）有效新结构数 / 测试生成数（论文口径；evaluator 同时给出原 eval_qa 的按串去重口径 `creativity_raw`）。
2. **开发集**：`test.json` 前 128 个测试探针，greedy，每 5000 步（与既有运行同一网格）。只用于选检查点与相位分析。
3. **检查点选择**：开发集 creativity 最大者，平局取最早，只在有检查点的步中选（`report.py`）。
4. **留出集**：`make_heldout.py --n 2048 --seed 20261011`（与训练/valid/test 全部 hash 不相交；sha256 记入日志），
   对每个运行的选中检查点**一次性** greedy 评测（`rescore.py --probe_file heldout.json`）。报告以留出集数值为准。
5. **次要指标**：validity、memo_own（对本臂实际训练配对，仅比较答案部分）、train_seen、nll_own 轨迹、终点、峰值、轨迹均值。
6. **成功判据**（C1/C2，对参照 `ntp_fixed` = 原论文设定）：按训练种子配对，留出集 creativity 差的**均值 ≥ 0.03 且 3 个种子全部 > 0**
   判 PASS；任一条件不满足为 FAIL；不足 3 对或含开发集数值为 INSUFFICIENT（`report.py --ref ntp_fixed --margin 0.03`）。
   margin 依据：留出集 2048 探针下单种子评测 SE ≈ 0.009；种子间差异的量级由首轮"近似重复"（0.063 validity）估计，
   0.03 + 全种子同号是保守门槛。**负结果同样按此表登记**（如 `setc_MN` 只回到 fixed 水平 = 不支持新方法）。

## 4. 每个 TL/setc 运行的诊断（`run_stage0b.sh diag`）
- churn：`diagnose.py --mode churn --data_dir`（相邻访问距离、P(perm_t = π_final)、承诺时间、结构级变化率、冻结排列统计）。
- gap：`diagnose.py --mode gap --model_dir "<run>/ckpts/step*"`——逐检查点（即按训练阶段）的 TL/AR 间隔、自身配对间隔、
  dropout 噪声尺度下的并列率与稳健率、TL/AR 决策一致率与遗憾（回答"早期是否由代理噪声播种"）、TL 交互项按位置占比。
- 训练日志：`train_log.jsonl`（每 50 步：loss/ntp/mtp、窗口内配对变化率）；scores 中 `nll_own` 可画"新颖性 vs 自身配对训练损失"
  （评审：若 setc 坍缩与 fixed 衰减落在同一曲线上，则 setc 只是把同一轨迹跑得更快）。

## 5. 预算与存储
A6000 实测（`runs/server-circle-40k/*/throughput.json`）：NTP 臂 ≈1 h/40k，TL/setc 臂 ≈2 h/40k（双流共卡）。
阶段 A ≈1 h；B ≈11.5 h（6 臂 ≈10 h + 锁定 0.3 h + 干预 1 h）；C ≈29 h（C1：2 种子 ×（1+2+2+2）h + setc_M05 3×2 h ≈20 h；C2：3 种子 ×（2+1）h ≈9 h）。合计 ≈41 GPU·时 < 余额 ≈48；若 B 的决策表指向停止，C 不执行。
存储：检查点 fp32 ≈0.5 GB × 8/运行；`run_stage0b.sh heldout` 完成留出评测后可 `PRUNE=1` 只保留选中检查点与 final_model。

> 维护修订（2026-10-16）：当前 launcher 已移除 `PRUNE` 删除路径；正式 `heldout` 只读冻结
> selection manifest，不删除 checkpoint。上行保留为旧计划记录，不作为当前操作指令。

## 6. 明确推迟（评审共识）
m=8、Sinkhorn / Gumbel 扰动匹配、`--regroup`、新鲜种子（IMLE 式）臂、更大模型——均在 B/C 指明"匹配本身"有作用之后再议；
代码均已就绪（`--m`、`--regroup`、`--random_tiebreak`）但不在本轮运行。

## 7. 提议的修订（**未签署**，需项目负责人批准后才进入预注册）：冲突分解

**动机。** 若阶段 B/C 中 `setc_MN`（匹配 TL / 名义 NTP）稳定，可至少有三种机制解释，而"分支目标冲突是一种正则"只是其一：
(i) 与**被匹配**的伙伴冲突才有用；(ii) 与**任意**伙伴冲突都行（匹配无关）；(iii) 与 TL 通道无关，只是每个 seed 有两个目标
（多目标/标签噪声式正则，已知现象）。另外，预注册比较对象是 `ntp_fixed`（无 TL）：若 `setc_MN` ≈ `ntp_fixed`，正确读法是
"冲突抵消了 TL 查表的害处"，而不是"冲突带来正则收益"；只有 `setc_MN` 按 §3.6 判 PASS 才支持后者。

| 标签 | 命令要点 | TL / NTP 目标 | 回答 |
|---|---|---|---|
| `tl_RN` | `setc_m4 --pair_rule fixed --fixed_pairing random --ntp_lambda 1` | 固定随机双射 / N | 无匹配的分支冲突 |
| `ntp_2tgt` | `ntp_fixed_grouped --pair_rule fixed --fixed_pairing random --ntp_lambda 0.5` | — / N + 固定随机伙伴 | 无 TL 的双目标 |

固定随机双射在每组内均匀抽取（独立随机流 seed+2），不动点约 25%，与 setc 冻结 π*（v2 实测 25.7%）可比。
`bash code/run_stage0b.sh conflict`；两臂各 3 种子。

| 结果（按 §3.6 规则，对 `setc_MN` 与 `ntp_fixed` 各比一次） | 解读 |
|---|---|
| `tl_RN` ≈ `setc_MN` | 匹配无关；起作用的是分支目标不一致——集合耦合作为方法的结论保持"失败" |
| `ntp_2tgt` ≈ `setc_MN` | TL 通道无关；现象归入多目标监督正则（需与已有文献对照，新意有限） |
| 二者都显著低于 `setc_MN` | 需要"被匹配的伙伴 + TL 通道"——唯一仍支持集合耦合有独立贡献的结果 |
| `setc_MN` ≈ `ntp_fixed` | 冲突只抵消 TL 的害处；"冲突正则"主张不成立 |

**预算（不增加总量的置换建议）。** 两臂合计 ≈9 GPU·时（tl_RN 3×2 h + ntp_2tgt 3×1 h）。建议以其替换 C 中的 `setc_M05`（3×2 h）与
`ntp_fixed_rep`（3×1 h）：λ=0.5 只在 `setc_MN` 已 PASS 时才有解读价值，NTP-only 回放只在 π* 回放显示效应时才需要；二者可顺延。
若团队预估的阶段 0 总量（≈67 GPU·时）超过 60 的上限，另可将 `tl_NN_rep` 顺延（−6 h），使总量回到上限内。

### 7.1 阶段 B 结果后的修订（2026-10-11，同样待批准）

**2×2（种子 42）**：只有 TL=M / NTP=N 稳定，NN、NM、MM 均坍缩。读法修订：
- **"名义"配对没有特殊地位**：datagen 把随机 hash 与随机 circle 配对，名义配对只是又一个任意双射。"TL 训固定随机、NTP 训名义"与
  "TL 训名义、NTP 训固定随机"在分布上**是同一个实验**（两个独立随机双射互换标签）。因此不能把结论表述为"TL 必须训非名义 / NTP 必须锚定名义"，
  也**不需要**再加反向的固定臂（与 `tl_RN` 等价）。
- **MN 与 NM 的真实不对称在匹配动力学**：Hungarian 跟随 TL 代价。MN 中 TL 训它自己的 argmin、NTP 训另一个固定配对 → 冲突持续；
  NM 中 TL 训名义 → TL 代价偏向名义 → π* 应收敛到恒等 → NM 退化为 NN。**可证伪预测**：`setc_NM` 的 `churn_analysis.json` 中
  `final_perm_stats.fixed_point_frac` 接近 1、`fixed_point_frac` 轨迹从 0.25 升向 1（对照：v1=MN 由 0.25 升到 0.32 后停住；v2=MM 稳在 0.25），
  且 NM 的 `train_memorization`（对名义）≈ memo_own ≈ 1。若成立，统一解释是"**两分支在收敛时仍训不同目标 ⇔ 稳定**"（H-TL 原式），
  方向性是动力学副产品；若 NM 的 π* 并未趋于恒等，方向性读法才获得独立支持。
- **锁定检验**：14k 时仍有约 0.3% 的访问在翻转（v2 实测 P(π_t=π_final)@14k = 0.998），锁定分支按设计压住这些翻转，
  不逐位相同是**预期结果**，不是随机流问题。等价检验应从 `state_step20000`（已完全固化）再做一次 2k 步两支（≈0.4 h）；
  若两支访问配对零差异，权重应逐位相同（`--deterministic`）。
- **晚期干预**：若 setc_MM 在 20k 仍处峰值附近（v2 的峰值正在 20k，25k 才明显下降），该结果是"**阻止**坍缩"而非"逆转"。
  如要主张逆转：从 `state_step20000` 在线续训并 `--state_at 30000` 到 30k（≈1 h），再从 30k 以 `--ntp_lambda 1` 分支（≈1 h）。
- MN 的 memo_own（对 π*）0.30 ≈ π* 不动点占比：AR 解码复现的是 NTP 训练的名义目标，TL 槽位承载 π*——两条通路在 MN 中分离，
  TL 对生成的作用是共享参数上的辅助损失。MN 是否优于 `ntp_fixed`（开发集峰值 0.250 vs 0.227，单种子、约 0.7 SE）只能由 §3.6 的留出集 3 种子判定。

**修订后的阶段 C（≈26 GPU·时，≤ 原 29）**：
必须：`ntp_fixed` s43/s44（2 h）、`setc_MN` s43/s44（4 h）、`tl_RN` ×3（6 h）、`ntp_2tgt` ×3（3 h）；
小项：锁定复检 @20k（0.4 h）、逆转分支 @30k（2 h）；坍缩格复现各 1 种子：`setc_MM` s43、`tl_NN` s43、`setc_NM` s43（6 h）。
顺延：`setc_M05`、`tl_NN_rep`、`ntp_fixed_rep`（两分支同一 π* 的回放按 H-TL 预测应坍缩，信息量低）。
