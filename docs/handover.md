# P6 项目交接文档

> 项目：Roll the dice 后续研究——配对多样性监督对种子条件生成的影响
> 交接日期：2026-10-16 · 交接人：ZCode（执行端）· 状态：事件 #9 修正已接受；Line 主检验已结束，2hash 进入独立确认候选阶段

---

## 0. 一页速览

**核心发现**（当前最可信的一句话）：

> 在当前已测试的配方、任务配置和评价程序中，改变训练时 hash→答案的监督配对图结构
> 显示出未见种子生成增益；**"同一答案挂多个 hash 入口"（ntp_fixed_2hash）**
> 在 Circle 与修复后的 Line 上均达到项目门槛，因此进入独立确认候选阶段。最初提出的
> 集合耦合方法（动态 Hungarian 匹配＋teacherless）的两个核心组件均被证明非必要。

**当前最大风险**：Line 任务的留出评价链曾因两个 bug（评价器用错任务＋加载终点模型而非
选中检查点）产出错误结论；已修复重评，但**修正后主 recipe（ntp_2tgt）未通过 Line 门槛**
——跨任务迁移主张已从"已确认"降级为"待定"。

**当前执行顺序**：
1. 完成发布收尾与评价身份登记（不重跑模型）；
2. 按配置审计结果冻结 Sibling 的数据、知识图、模型初始化和评价协议；
3. 仅在门禁通过后预注册并运行 fixed／b128／2hash 三臂确认；scale 继续延期。

---

## 1. 项目定位与背景

### 基础论文

**"Roll the dice & look before you leap"** (Nagarajan et al., ICML 2025 杰出论文,
arXiv:2504.15266)：在受控任务（Circle/Line Construction, Sibling/Triangle Discovery）上
研究语言模型的"算法创造力"。发现：① next-token 训练的短视性；② seed-conditioning
（训练时把随机前缀与答案任意配对）能让模型学会"换种子换答案"。**论文自认不理解任意配对
为何泛化（§B.1）——这是我们的切入点。**

### 我们的问题演化

| 阶段 | 问题 | 结论 |
|---|---|---|
| 最初 | 动态集合耦合（Hungarian 匹配在 teacherless 代价上）能否超越固定配对？ | setc_MN 预注册 PASS（Circle +0.047），但匹配与 teacherless 均非必要 |
| 中期 | 增益的承重结构是什么？ | 配对多样性监督图结构；batch 扩大不充分；模型仍记忆训练配对（"查什么"变了） |
| 当前 | 最优配对形式是否跨任务一致？ | 2tgt 在 Circle 最优但 Line 收窄；2hash 在两任务均达标——任务依赖或不依赖 |

### 研究对象表述（第九轮评审建议）

> **配对多重性如何影响种子条件生成：监督结构、计算分配与未见结构产出。**
> 研究框架：hash—答案**监督二部图**（fixed=一对一；2tgt=每 hash 双目标；2hash=答案多入口
> 每 hash 单目标；cycle-k=渐密图）。

---

## 2. 实验与结论完整清单

### 2.1 阶段 0：Circle 任务基础对照（2026-10-09～11，gpu-gfkd A6000）

**五臂对照**（各 40k 步，3 种子）→ **2×2 分解**（TL×NTP 目标）→ **预注册 PASS**：

| 比较 | 留出 creativity | 判定 |
|---|---|---|
| setc_MN vs ntp_fixed | +0.047（3/3 正，≥0.03） | **PASS（预注册）** |
| (TL=N,NTP=N) / (M,M) / (N,M) | 全部坍塌至 ~0.02–0.06 | 只有 (M,N) 方向稳定 |
| 晚期干预（坍塌中切 NTP 目标） | 0.023→0.227 | 恢复生成的因果证据 |

### 2.2 阶段 0b：机制对照（2026-10-11～12）

| 臂 | Circle 留出 | 回答 |
|---|---|---|
| tl_RN（固定随机配对＋TL） | 0.233 | Hungarian 匹配**非必要** |
| **ntp_2tgt（纯 NTP 双目标）** | **0.294** | teacherless **非必要**；全场最优 |
| ntp_fixed_b128 | 0.165 | 纯 batch 扩大**不充分** |

### 2.3 阶段 0c：multiplicity（2026-10-13～15，gpu-lzn L40）

| 配方 | Circle 留出 | Δ vs fixed | §4.1b 判定 |
|---|---|---|---|
| ntp_2tgt | 0.294 | +0.132 | PASS（vs b128 +0.129, vs 2hash +0.053） |
| ntp_2tgt_sample（等计算） | 0.228 | +0.066 | PASS |
| ntp_fixed_2hash（等对数） | 0.241 | +0.079 | PASS |
| b128 | 0.165 | +0.003 | FAIL（零效应对照确认） |
| k=3 / k=4 | 0.200 / 0.176 | — | k=2 均值峰值（非逐种子单调、非等计算） |

### 2.4 Phase 1：Line 跨任务（2026-10-15～16，事件 #9 修复后）

| 配方 | Line 留出 | Δ vs fixed | 判定 |
|---|---|---|---|
| ntp_fixed | 0.155 | — | — |
| b128 | 0.157 | +0.002 | 平均差约 +0.002，未达到项目门槛 |
| **ntp_2tgt（预指定主 recipe）** | 0.169 | **+0.014** | **未过门**（3/3 正但 <0.03） |
| ntp_2tgt_sample | 0.160 | 高方差 | 不稳 |
| **ntp_fixed_2hash** | **0.190** | **+0.034** | **PASS（辅助臂）** |

---

## 3. 遇到的问题：完整事件登记（#1–#9）

| # | 事件 | 影响 | 处置 |
|---|---|---|---|
| 1 | hybrid 数据双重展开（teacherless 臂全零） | tl_hybrid 作废 | 修复 train.py 消费逻辑 |
| 2 | （预留） | — | — |
| 3 | NTP 前向误取整元素（randperm 臂从未真正重配） | G1 对照作废 | 修复前缀/答案分离取值 |
| 4 | evaluator seen() 尾部 pad 不截断 | 批量生成后已见解误判新解 | 修复＋回归测试 |
| 5 | review_analyses.py 含 hash 前缀比较 | "非过拟合"错误结论 | 独立核验撤回；答案段比较修正 |
| 6 | 服务器拉分支失败→conflict 臂静默跳过 | 阶段 C 不完整（后补齐） | manifest 驱动完成判定 |
| 7 | tl_RN 配对规格偏差（random vs 签名要求的 derangement） | 24% 种子无冲突 | unif 变体保留为剂量点；按签名规格重跑 |
| 8 | report.py 按标签聚合混入 unif 运行（n=4 污染） | 表格不可引用 | 重复守卫＋目录标签；清洁 n=3 报告 |
| **9** | **Line 留出评价链双错：Circle 规则评 Line ＋ 加载 final_model 而非选中 ckpt** | **增补 12 全部作废**；主 recipe 从 PASS 降为未过门 | rescore.py 加 line＋唯一计数统一；按冻结 selection.tsv 重评；旧结果保留 |

**方法论教训**（已固化为仓库约定）：
- "预定运行集合 = 已验证完成集合"驱动完成判定，不以"无 FAIL 即完成"代替；
- `--step` 仅是标签，必须核对实际加载的 `--model_dir`；
- 评价器任务枚举必须与训练接口同步（rescore.py 漏加 line 是事件 #9 根因之一）；
- 增补式记录（不删旧更正）——增补 8→9→10→11→13 的撤回链条完整保留。

---

## 4. 代码与评价基础设施

### 4.1 关键文件

| 文件 | 职责 |
|---|---|
| `code/train.py` | 统一训练器：五预设臂（ntp_fixed / setc_m4 / tl_fixed_m4 等）＋ `--pair_rule multi` k 目标＋ `--target_mode weighted/sample`＋`--fixed_pairing random/derangement/cycle`＋`--lock_at`＋`--pairing_file` 回放＋全状态断点续训 |
| `code/evaluator.py` | 三任务（circle/sibling/line）有效性＋seen＋creativity（canonical 去重＋raw 双口径）；`--selftest` 35+ 用例 |
| `code/rescore.py` | 留出集重评（**事件 #9 后 task 必须显式传入**）；核对 checkpoint/probe 身份并保存原始预测 `preds_stepN.jsonl` |
| `code/report.py` | 聚合＋按种子配对差＋预注册 verdicts（margin/min_seeds 可配）；`--label_by dir` 防标签污染 |
| `code/datagen.py` | 固定种子数据生成（circle/sibling/line，含 path_mode） |
| `code/make_variant_data.py` | dup_hash 变异（2hash 臂的数据） |
| `code/make_heldout.py` | 留出集生成（SHA256 存档） |
| `code/run_stage0b.sh` | 一键阶段启动器（A/B/conflict/multiplicity/heldout/report） |
| `code/test_pipeline.py` | 35 项回归测试（含变异检查） |

### 4.2 预注册评价协议

1. 训练时每 5k 步评开发集（128 test 探针，greedy）；
2. 训练完生成留出集（`make_heldout.py`，2048 全新 hash，排除训练/开发已用，SHA256 存档）；
3. 开发集最高 creativity 的检查点选中（并列取早），`selection.tsv` 冻结落盘；
4. 选中检查点→留出集一次评价（`rescore.py --task <task> --model_dir ckpts/stepN --step N`）；
5. `report.py --ref ntp_fixed --margin 0.03 --min_seeds 3` 出机器 verdicts。

---

## 5. 当前仓库与资源状态

### 5.1 仓库（https://github.com/StephCpa/RolltheDice）

- **main** 最新：`1f967d4`（全部 13 个增补＋修正版 work-summary＋全部运行记录已入库）
- 分支 `claude/zen-bohr-gq1si2`：合作者（另一位 AI）的工具链贡献，已全部合入 main
- 关键路径：`runs/stage0-report.md`（全部增补1–13）、`docs/work-summary.md`（面向阅读的总结）、
  `plan/`（已签协议与修订）、`runs/line-phase1/`（最新一组实验＋修复后 heldout 分数）

### 5.2 服务器

| 服务器 | 配置 | 环境 | 用途 |
|---|---|---|---|
| gpu-gfkd | 8×A6000 48GB（共享） | torch 2.4.0+cu121 / tf 4.47.1 | 阶段 0 全部运行 |
| gpu-lzn | 2×L40 48GB（共享） | 同上（~/envs/dllm_cf_l40_py312） | multiplicity＋Line Phase 1 |
| 本地 | RTX 2080 Ti 11GB | torch 2.6.0+cu124 / tf 5.19（**与服务器不同栈**） | 烟雾测试＋分析脚本 |

SSH 配置均在 `~/.ssh/config`（跳板机 118.252.19.90:1722，间歇性断连属正常，重试即可）。

### 5.3 预算

- 阶段 0 上限 60 A6000·h：已用约 58h（触顶）
- ICML 2027 计划（~890 GPU·h，已批，NeurIPS 顺延预案）——Phase 1 Line 已花约 24 L40·h
- 后续大额支出需审阅方确认

---

## 6. 接手后最优先的三件事

### ① 事件 #9 修复结果（已接受）

- 核对 `runs/line-phase1/*/s*/heldout/scores.jsonl`（修正版）与 `heldout_wrong_circle_final/`（事件证据）；
- 确认修正后判定：主 recipe 2tgt 未过门（+0.014），2hash 辅助臂过门（+0.034）；
- 项目负责人已接受该判定：不再为 `ntp_2tgt` 追加 Line 追门槛；保留 2hash 为下一轮独立确认候选。

### ② 若确认 2hash 为下一候选：起草预注册修订

2hash 在当前已测试配置中同时达到两任务门槛（Circle +0.079、Line +0.034），但其增益机制
（更多训练 hash？更多配对数？答案多入口本身？）尚未与 b128 完全分离——Circle 上
2hash−b128=+0.076（同 128 序列纯 NTP），这是正面对照，但 Line 上 2hash−b128=+0.032，
幅度不一致。预注册应指定 2hash vs b128 为主比较。

### ③ Sibling 配置核对（已完成静态核对，训练仍暂缓）

审阅方指示：先核对原论文/脚本/当前实现的数据规模差异（原 `sibling.5.500` vs 我们
datagen 的 `10.1000`）、初始化、训练步数（原 800k 步），**不直接沿用旧上限核价**。

核对记录见 [`docs/sibling-config-audit-20261016.md`](sibling-config-audit-20261016.md)。
记录确认：官方可追溯 artifact 为 `sibling.5.500.10.50000`；当前本地 `10×1000`
数据不能由现有 `datagen.py` 再生，且缺少原评价所需的完整知识图。原始脚本的
8-layer fresh-init／800k 与当前 P6 trainer 的 pretrained GPT-2 配置也不能混用。
在数据、知识图、初始化、步数和评价身份全部冻结前，不启动九个 Sibling 运行。

---

## 7. 常见陷阱（接手者必读）

1. **旧版 `--step` 只是标签**——事件 #9 的实际模型由 `--model_dir` 决定；当前 rescore.py 要求它与 `ckpts/stepN` 或终点身份一致。
2. **新任务必须同步三处**：`datagen.py` 生成＋`evaluator.py` 有效性＋`train.py` 和 `rescore.py`
   的 `--task` choices——漏一处就会静默用默认任务评价（事件 #9 根因）。
3. **报告聚合用 `--label_by dir`**——不用会按 run_meta 标签聚合，unif/derangement 变体会混入。
4. **留出集看过就登记**——事件 #9 修复后，Line 留出集已"重复使用"，不能重新宣称首次查看。
5. **服务器拉分支可能静默失败**——`git fetch` 断连后 checkout 不报错但拿到旧代码（事件 #6/7
   根因）；部署代码后用 `grep` 确认关键改动在位。
6. **GPU 上的同事任务**——不要 kill 任何非本项目的进程；GPU 4/7 曾被他人占满导致 OOM。
7. **跨环境比较**——本地 2080Ti 与服务器 L40/A6000 的数字不直接比；同环境内的臂间差才可靠。
8. **增补式写作**——不要删旧增补（增补 12 作废但保留）；新更正追加新编号。

---

## 8. 关键人物与角色

| 角色 | 职责 |
|---|---|
| 项目负责人（用户） | 预算批准、预注册签署、方向决策 |
| 执行端（ZCode，本文档作者） | 代码实现、实验运行、分析、归档推送 |
| 外部评审（人类，共 10 轮） | 方案审查、数值核验、纪律监督——**每轮意见全部落地为增补** |
| 分支合作者（AI） | 工具链贡献（config-driven trainer、report/selection 协议、35 项测试等） |

---

## 9. 文件索引

```
RolltheDice/
├── docs/work-summary.md          ← 面向阅读的全局总结（已含修正后 Line 数字）
├── docs/handover.md              ← 本文档
├── plan/stage0-plan.md           ← 原阶段 0 计划
├── plan/stage0b-amendment-1.md   ← 预注册修订（含 §4.1b 签署记录）
├── plan/icml2027-plan.md         ← ICML 2027 投稿计划（~890 GPU·h）
├── runs/stage0-report.md         ← 全部实验增补 1–13（含事件登记与撤回链）
├── runs/line-phase1/             ← Phase 1 Line 全部产物（含修复后 heldout）
├── runs/stage0b/                 ← 阶段 0b/0c 产物
├── runs/server-circle-40k/       ← 阶段 0 服务器运行
├── code/                         ← 全部代码（见 §4.1）
└── data/                         ← .gitignore 排除，由 datagen.py 再生
```

---

## 10. 发布收尾记录（2026-10-16）

本轮已完成且均为 development 记录：

1. [`runs/line-phase1/report.md`](../runs/line-phase1/report.md) 与 [`report.json`](../runs/line-phase1/report.json)
   按冻结 [`selection.tsv`](../runs/line-phase1/selection.tsv) 从现有 15 条修正后的 Line 留出记录重出；`ntp_2tgt` 为 FAIL（均值 +0.013835），
   `ntp_fixed_2hash` 为 PASS（均值 +0.034342）。没有重新生成模型输出。
2. 旧的 Circle 规则／`final_model` 报告保存在
   [`report_event9_void_circle_final.md`](../runs/line-phase1/report_event9_void_circle_final.md)，明确标记为作废历史版本。
3. [`code/index_evaluation_identity.py`](../code/index_evaluation_identity.py) 生成只读身份索引
   [`evaluation_identity_20261016.json`](../runs/line-phase1/evaluation_identity_20261016.json)；
   缺少 `rescore_meta.json` 的旧 scores 只保留作证据，不自动升级为可复用的正式结果。
4. `rescore.py` 现在要求显式 `--task` 与一致的 checkpoint 身份；`run_stage0b.sh` 的
   `select` 与 `heldout` 分离，正式评价只读冻结清单并核对任务、模型目录、评价器和探针身份。

## 11. 当前门禁后的下一步

下一轮只保留 `ntp_fixed`、`ntp_fixed_b128`、`ntp_fixed_2hash` 三臂和三个预先冻结的训练种子；
核心比较为 `2hash−b128`，同时报告 `2hash−fixed`。不加入 `2tgt`、sample、k 扫描、
Hungarian、teacherless、新正则或 scale。上述内容是设计约束，不是已经产生的实验结果。

*本文档由执行端生成于 2026-10-16，对应 main@`1f967d4`；本轮状态修订与收尾记录追加于同日。
如有疑问请对照仓库原始文件和增补记录，而非仅依赖本文档摘要。*
