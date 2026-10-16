# RolltheDice · P6 研究代码与实验日志

ICML 2025 杰出论文 *Roll the dice & look before you leap*（arXiv:2504.15266）的后续研究项目 **P6**：**集合耦合的多解生成训练**（set-coupled multi-solution training）——训练时在 m 个种子与 m 个已验证解之间用 Hungarian 匹配选择当轮最优双射，利用 teacherless（并行预测）接口使 m² 个配对代价只需 m 次前向。

**状态：阶段 0（限额原型）进行中。所有结果为 development 状态。**方法论框架见 [Meta-method](https://github.com/StephCpa/Meta-method)（本项目为其 P6 工作包，方案分析与决策记录在 Meta-method 仓库的 `P6-Roll_the_dice/` 下，本仓库承载代码与实验日志）。

## 仓库结构

| 目录 | 内容 |
|---|---|
| `code/` | 训练器（组级臂由 `GroupCfg` 配置：配对规则 × TL 目标 × NTP λ × TL 权重；锁定 / π* 回放 / 完整状态续训）、数据生成、评价器（自检 17/17）、诊断（churn / gap / export_pairing / compare / mi / readout）、C01 基准、留出集 `make_heldout.py`、重评 `rescore.py`、汇总 `report.py`、阶段 0b 启动器 `run_stage0b.sh`、回归测试 `test_pipeline.py`（27 项） |
| `runs/` | 实验日志：逐步评价 `scores.jsonl`、吞吐、churn 记录（gzip）、失效/偏差登记 |
| `sources/` `analysis/` `verification/` `plan/` | 原始研究方案存档、Meta-method 分析、查新核验记录、阶段 0 计划与**阶段 0b 执行方案（含预注册）`plan/stage0b-plan.md`** |

## 复现

```bash
pip install -r requirements.txt
python code/evaluator.py --selftest          # 评价器自检
python code/test_pipeline.py                 # 语义回归测试（CPU、离线，微型随机 GPT-2）
python code/datagen.py --task circle --out data   # 数据生成（固定种子 20261009）
python code/train.py --arm ntp_fixed --task circle \
  --data_dir data/circle.10.9.10.10000 --out runs/x --steps 40000 --save_final
python code/make_heldout.py --data_dir data/circle.10.9.10.10000 --n 2048   # 留出探针集（与全部已用 hash 不相交）
python code/rescore.py --model_dir runs/x/final_model --data_dir data/circle.10.9.10.10000 \
  --task circle --probe_file data/circle.10.9.10.10000/heldout.json \
  --out runs/x/heldout --step 40000   # 重评 / 留出评测
TASK=circle bash code/run_stage0b.sh B        # 阶段 0b：A|B|lock|intervene|C|replay|diag|select|heldout|report
# selection 生成与正式留出评价分开；heldout 只读已有 selection.tsv
TASK=circle bash code/run_stage0b.sh select
TASK=circle bash code/run_stage0b.sh heldout
```

臂：`ntp_noseed`（无种子）｜`ntp_fixed`（固定配对，原论文设定）｜`ntp_randperm`（持久种子集＋每轮随机双射，G1 对照）｜`tl_hybrid`（原 teacherless 1:3 混合）｜`setc_m4`（集合耦合原型）｜`tl_fixed_m4` / `tl_randperm_m4`（与 setc 同目标、同算力、同批组织，仅配对规则为恒等 / 随机——分离"最优配对"效应的等算力对照）。另有 `ntp_fixed_grouped`（组批组织版 ntp_fixed，流水线自检）。组级臂可用 `--pair_rule/--tl_target/--ntp_lambda/--mtp_weight` 覆盖预设：`setc_m4 --ntp_lambda 1` 精确重建偏差版 v1（匹配 TL / 名义 NTP），`setc_m4 --tl_target nominal` 为反向组合。干预与回放：`--state_at`/`--resume_from`（逐位续训）、`--lock_at`（冻结-锁定）、`--pairing_file`（π* 回放；`runs/*/setc_m4*/pistar_final.json` 已从 churn 导出）、`--ckpt_every`（检查点选择）。其他开关：`--random_tiebreak`、`--regroup`、`--m 64 --groups_per_step 1`（整批匹配）、`--deterministic`。训练语义对照原论文代码快照 `chenwu98/algorithmic-creativity@7176dc7` 逐条重实现（该栈依赖 2021 版 transformers，无法在现代 Python 运行）。

## 主要发现（截至 2026-10-10，详见 runs/stage0-report.md）

1. 无 seed 臂模式塌缩复现（唯一有效输出）；seed 臂在未见 hash 上产生新有效解。（评审注：greedy＋恒定 prompt 下"唯一输出"必然成立，尚需温度采样对照，见评审 §1.3。）
2. setc 首轮（含偏差，见下）在小模型单种子上小幅领先各生成指标。
3. **配对动力学三阶段**：高变化期（churn≈随机置换期望 1.917）→ 变化减少期 → 观测冻结期（~12k 步起 churn=0）。最优配对训练收敛到内生选择的固定配对。
4. **服务器修正轮（同环境，单种子）**：修正后的 setc_v2 记住自选配对后坍缩（终点 creativity 0.023），低于 ntp_fixed_v2（0.180）；"集合耦合已有生成优势"撤回，第 2 条仅指偏差版。当前首要待检解释是 teacherless 损失对 hash 的查表效应（TL 与 NTP 目标一致时可被完全拟合），按 TL 目标 × NTP 目标 2×2 检验——见 `plan/stage0b-plan.md`。

## 事件登记（完整列表见 runs/README.md）

- **#1** tl_hybrid 首轮失效：hybrid JSON 被二次展开 → `tl_hybrid_INVALID-doubleexpansion/`
- **#3** NTP 配对 bug：group_step NTP 前向误取整元素 → randperm 首轮从未真正重配（`ntp_randperm_INVALID-nominal-ntp/`）、setc 首轮 NTP 分量训名义配对（`setc_m4/DEVIATION.md`）。已修复，重跑进行中。
- **#4** 评价 `seen()` 未截断 `</a>` 后的批量 pad → circle 的 test_seen 偏低、test_unseen / 唯一新解偏高（validity、memo、唯一有效不受影响）。已修复并加回归测试；已有模型需用 `code/rescore.py` 重评。
- **#5** `review_analyses.py` 的 memo 重评分比较整串（含 hash 前缀）→ memo_πfinal 只能命中不动点；报告"增补 2"中"setc_v2 只复现 27% 训练对"的裁定作废（35/35 个不动点探针均被复现）。已修复；evaluator 新增 `memo_own`。

外部评审（代码核对、证据强度、对照设计与后续运行计划）见 [`analysis/2026-10-10-external-review.md`](analysis/2026-10-10-external-review.md)。

## 许可与引用

研究用途；原论文版权归其作者。引用本仓库代码时请注明项目 P6 与原论文 arXiv:2504.15266。
