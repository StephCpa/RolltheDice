# RolltheDice · P6 研究代码与实验日志

ICML 2025 杰出论文 *Roll the dice & look before you leap*（arXiv:2504.15266）的后续研究项目 **P6**：**集合耦合的多解生成训练**（set-coupled multi-solution training）——训练时在 m 个种子与 m 个已验证解之间用 Hungarian 匹配选择当轮最优双射，利用 teacherless（并行预测）接口使 m² 个配对代价只需 m 次前向。

**状态：阶段 0（限额原型）进行中。所有结果为 development 状态。**方法论框架见 [Meta-method](https://github.com/StephCpa/Meta-method)（本项目为其 P6 工作包，方案分析与决策记录在 Meta-method 仓库的 `P6-Roll_the_dice/` 下，本仓库承载代码与实验日志）。

## 仓库结构

| 目录 | 内容 |
|---|---|
| `code/` | 干净重实现的训练器（五条原臂＋两条等算力对照）、数据生成、评价器（自检 15/15）、吞吐基准、诊断脚本、重评脚本 `rescore.py`、回归测试 `test_pipeline.py` |
| `runs/` | 实验日志：逐步评价 `scores.jsonl`、吞吐、churn 记录（gzip）、失效/偏差登记 |
| `sources/` `analysis/` `verification/` `plan/` | 原始研究方案存档、Meta-method 分析、查新核验记录、阶段 0 计划 |

## 复现

```bash
pip install -r requirements.txt
python code/evaluator.py --selftest          # 评价器自检
python code/test_pipeline.py                 # 语义回归测试（CPU、离线，微型随机 GPT-2）
python code/datagen.py --task circle --out data   # 数据生成（固定种子 20261009）
python code/train.py --arm ntp_fixed --task circle \
  --data_dir data/circle.10.9.10.10000 --out runs/x --steps 40000 --save_final
python code/rescore.py --model_dir runs/x/final_model \
  --data_dir data/circle.10.9.10.10000 --out runs/x/rescore --n_probes 2048   # 重评已有模型
```

臂：`ntp_noseed`（无种子）｜`ntp_fixed`（固定配对，原论文设定）｜`ntp_randperm`（持久种子集＋每轮随机双射，G1 对照）｜`tl_hybrid`（原 teacherless 1:3 混合）｜`setc_m4`（集合耦合原型）｜`tl_fixed_m4` / `tl_randperm_m4`（与 setc 同目标、同算力、同批组织，仅配对规则为恒等 / 随机——分离"最优配对"效应的等算力对照）。新开关：`--random_tiebreak`（Hungarian 并列随机打破）、`--regroup`（每轮重新分组）、`--m 64 --groups_per_step 1`（整批匹配）。训练语义对照原论文代码快照 `chenwu98/algorithmic-creativity@7176dc7` 逐条重实现（该栈依赖 2021 版 transformers，无法在现代 Python 运行）。

## 主要发现（截至 2026-10-10，详见 runs/stage0-report.md）

1. 无 seed 臂模式塌缩复现（唯一有效输出）；seed 臂在未见 hash 上产生新有效解。（评审注：greedy＋恒定 prompt 下"唯一输出"必然成立，尚需温度采样对照，见评审 §1.3。）
2. setc 首轮（含偏差，见下）在小模型单种子上小幅领先各生成指标。
3. **配对动力学三阶段**：高变化期（churn≈随机置换期望 1.917）→ 变化减少期 → 观测冻结期（~12k 步起 churn=0）。最优配对训练收敛到内生选择的固定配对。

## 事件登记（完整列表见 runs/README.md）

- **#1** tl_hybrid 首轮失效：hybrid JSON 被二次展开 → `tl_hybrid_INVALID-doubleexpansion/`
- **#3** NTP 配对 bug：group_step NTP 前向误取整元素 → randperm 首轮从未真正重配（`ntp_randperm_INVALID-nominal-ntp/`）、setc 首轮 NTP 分量训名义配对（`setc_m4/DEVIATION.md`）。已修复，重跑进行中。
- **#4** 评价 `seen()` 未截断 `</a>` 后的批量 pad → circle 的 test_seen 偏低、test_unseen / 唯一新解偏高（validity、memo、唯一有效不受影响）。已修复并加回归测试；已有模型需用 `code/rescore.py` 重评。

外部评审（代码核对、证据强度、对照设计与后续运行计划）见 [`analysis/2026-10-10-external-review.md`](analysis/2026-10-10-external-review.md)。

## 许可与引用

研究用途；原论文版权归其作者。引用本仓库代码时请注明项目 P6 与原论文 arXiv:2504.15266。
