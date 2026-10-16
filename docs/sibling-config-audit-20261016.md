# Sibling 配置核对记录（development）

> 核对日期：2026-10-16（Asia/Shanghai）  
> 范围：发布收尾前的 Sibling 三臂确认准备。只检查仓库材料、随仓库保留的上游源码快照和公开数据元信息；本记录没有启动训练、没有读取服务器权重，也没有产生新的模型结果。

## 1. 来源、版本和核验动作

本轮核对了以下材料：

| 来源 | 定位 | 核验动作 |
|---|---|---|
| P6 canonical repo | `main@f089a51` | 读取 `docs/handover.md`、`plan/icml2027-plan.md`、`code/datagen.py`、`code/evaluator.py`、`code/train.py`；检查 Sibling 配置和当前 launcher/评价接口。 |
| 上游代码快照 | `external/algorithmic-creativity`（P6 数据目录随附的快照；P6 代码注释标为 `@7176dc7`） | 对照 `sibling-discovery/ntp/sibling.ipynb`、`sibling_no_hash.ipynb`、`teacherless/sibling_hybrid.ipynb`、`ntp/train.sh`、`ntp/run_train.sh`、`ntp/run_eval.sh`。 |
| 公开数据元信息 | [ChenWu98/sibling.5.500.10.50000](https://huggingface.co/datasets/ChenWu98/sibling.5.500.10.50000)；上游 [README](https://github.com/chenwu98/algorithmic-creativity) | 核对数据集名称、论文链接、split 行数和 README 的示例链接。 |
| 当前本地数据 | `P6-Roll_the_dice/data/sibling.10.1000.*.50000` | 读取 JSON 行数、实体编号范围、hash 唯一性和 SHA-256；不修改文件。 |

未覆盖：服务器上的原始数据目录、权重、论文补充材料中未随仓库保留的配置、当前机器的 GPU 预算实测。公开数据页面是外部元信息，不等于本地下载文件已经冻结。

## 2. 已确认的配置冲突

### 2.1 数据规模和数据来源

有两套相互冲突的上游线索：

* `sibling.ipynb` 的直接 hash 数据生成单元格写出 `NUM_A = 10`、`NUM_B_PER_A = 1000`、`HASH_STR_LEN = 10`，随后在 downsampling 单元格把训练集抽到 `training_size = 50000`，测试探针为 3000。
* 官方 README 的 Sibling 数据示例、`ntp/run_train.sh`/`run_eval.sh`、`sibling_no_hash.ipynb` 和 teacherless notebook 使用 `sibling.5.500.10.50000`。公开 Hugging Face 数据卡同名，train 显示 50k 行、总行数 53k，并链接论文 arXiv:2504.15266。

因此，`5×500` 是当前可追溯的论文数据 artifact 配置；`10×1000` 只能登记为上游 notebook 与脚本不一致的开发候选，不能未经裁决就作为“原配置”。

通过 [Hugging Face 数据集 API](https://huggingface.co/api/datasets/ChenWu98/sibling.5.500.10.50000)（2026-10-16 查询）可进一步固定：dataset revision 为 `edb56d3aba44007a926f1b3535812009b28934e4`，features 只有 `input_text`、`target_text`，split 为 `train=50000`、`valid=3000`；公开 artifact 没有随 parquet 提供 `entities_b1_dict.json`、`entities_b2_dict.json` 或 `test.json`。这既支持 `5×500` 的数据规模，也意味着完整知识图仍需从获准的源材料单独冻结，不能假定由 HF split 自动提供。

本地的 `sibling.10.1000.10.50000/train.json` 有 49,766 条记录，b1/b2 编号覆盖 0–9999；`valid.json` 有 3000 条，`test.json` 有 6000 条（3000 train probes + 3000 test probes）。该形状与 notebook 的“先从全量组合抽 50,000，再按 0.5% 留出”相符，但当前 `code/datagen.py` 的 `write_split` 传入 `len(train_sequences)`，重跑会写出约 995 万条训练样本，不能再生这份本地文件。现有数据不能当作当前生成器的可复现输出。

当前本地文件哈希（仅作输入身份记录）：

| 文件 | SHA-256（前缀） | 行数 |
|---|---:|---:|
| `sibling.10.1000.10.50000/train.json` | `142d92c186724f9230859f6bf42112318116a4a944d0e43c3a76fec9a917712f` | 49,766 |
| `sibling.10.1000.10.50000/valid.json` | `f17ffaa74e27e10c0fe7fce02ddd5a2401aedd4073b18fc120ea67eca1640fa` | 3,000 |
| `sibling.10.1000.10.50000/test.json` | `e9e9aba5d365f6de9ca2ba4afc7e4c985c8f5c3ca7f2715f6af8d25972e8c85` | 6,000 |
| `sibling.10.1000.0.50000/train.json` | `2464fd5447d74d61bb7446a012aa7d0c79ea9fb212a1d9a362fce0d9f42566c0` | 49,766 |

### 2.2 知识图和有效性口径

上游 `sibling-discovery/ntp/eval_qa.py` 从 `entities_b1_dict.json` 和 `entities_b2_dict.json` 读取完整知识图，再判定 `(b1,a)` 和 `(b2,a)` 是否为真边。当前 `code/evaluator.py` 的 `build_sibling_graph()` 从 `train.json` 重建边集合；当前本地数据也没有随目录保存上述两个 dict 文件。这是评价语义差异，不能用“训练抽样几乎覆盖全部边”替代契约确认。

对现有 `sibling.10.1000.10.50000` 的静态计数显示，train-derived 图每个 `a` 实际只覆盖 990–997 个 b1 边、988–996 个 b2 边，合计缺少 66 个 b1 边和 69 个 b2 边（无 hash 版本相同）。因此该口径确实会把一部分知识图真边判成无效，影响 validity/creativity，而不是仅有命名差异。

启动前必须选定并冻结以下一种口径：

1. 使用官方/再生数据附带的完整 `entities_b1_dict.json`、`entities_b2_dict.json`，评价器显式读取并记录两个文件哈希；或
2. 明确把 train-derived graph 登记为本轮新评价口径，另写适用边界，不能称作原论文评价的直接复现。

当前建议采用第 1 种；在图文件缺失时暂停，不静默回退到 train-derived graph。

### 2.3 模型初始化和训练长度

上游 NTP `train.sh` 固定 `gpt2`、`--init_weights`、`--n_layer 8`、bs 64、lr `1e-4`、wd 0、seq/max/block 128、constant schedule with warmup、800,000 steps、fp16；teacherless 脚本为 1,200,000 steps。当前 P6 `code/train.py` 直接 `GPT2LMHeadModel.from_pretrained()`，没有 `init_weights`/`n_layer` 参数；现有 Circle/Line runs 也都采用该当前 trainer。两者不能混写成同一个复现配置。

启动前必须登记研究对象：

* **严格原任务复现**：实现并冻结 8-layer fresh-init、800k（TL 为 1.2M）等上游参数；结果只用于原任务复现。
* **同系统 Sibling 确认**：沿用当前 P6 trainer/模型栈以隔离任务迁移；明确这是 P6 系统确认，记录与上游初始化及步数的差异。

本轮用户决定的最小三臂只允许在其中一个配置下运行，不能臂间切换。

## 3. 三臂确认的待冻结契约

以下是用户已经指定的臂和比较关系；数值结果尚未产生：

| 臂 | 数据/批量角色 | 必须保持相同 |
|---|---|---|
| `ntp_fixed` | 基础数据，一对一 hash→答案，原始 batch 参照（建议保留 bs=64 身份） | 模型、优化器、数据快照、训练 seed 集、步数协议、checkpoint 选择和评价器版本 |
| `ntp_fixed_b128` | 基础数据，一对一配对，128 序列/步；作为固定配对@128 的计算对照 | 除 batch/有效 token 暴露外全部相同 |
| `ntp_fixed_2hash` | `dup_hash(k=2)`：每个原训练答案增加一个新 hash，每个 hash 仍只有一个目标；128 序列/步 | 除数据变体及其明确的样本/ token 暴露外全部相同 |

核心比较为 `2hash − b128`，保留 `2hash − fixed`。三臂均使用三个预先冻结的独立训练 seed（建议沿用 42/43/44），但不得把本轮开发阶段的旧运行当作 Sibling 确认重复。

`dup_hash` 必须在基础数据冻结后生成：新 hash 与基础 train/valid/test/heldout 全部不相交，记录变体 seed、基础数据 SHA、变体 train SHA、行数和 token 统计；variant 的 valid/test/heldout 应逐字复制基础数据。每臂实际序列数、token 数、更新步数和墙钟成本分别记录，不能仅以 batch size 代替计算量。

评价协议在启动前写死：

1. 任务身份为 `sibling`，训练、重评和报告的元数据都显式记录；缺失或冲突时报错。
2. 开发集检查点按最高 canonical creativity、并列取早选择；选择表只读使用，不在正式评价阶段重生成。
3. 新建 Sibling 留出 hash，记录生成器版本、排除集合、SHA 和生成日期；留出集读数只评价一次。
4. 选中 checkpoint 的模型目录、checkpoint 身份、评价器/探针版本和完整知识图身份同时核对；不能只看 `step` 标签。
5. 报告同时列出 canonical/raw 去重、validity、seen 和连续的按 seed 差值；既定 margin 是否适合 Sibling 指标尺度，应在读数前登记。

## 4. 启动前门禁

以下项目任一未完成，都不启动九个训练运行：

- [ ] 选择并下载/冻结 `5×500` 论文 artifact，或完成对 `10×1000` notebook 候选的正式裁决；保存文件 SHA 和 split/schema 清单。
- [ ] 保存完整 Sibling 知识图（或书面批准新的 train-derived 口径）并让 evaluator 显式读取。
- [ ] 冻结“严格原任务”或“同系统确认”模型初始化/层数，不能沿用默认值而不记录。
- [ ] 冻结步数、batch、warmup、序列长度、seed、checkpoint 间隔和预算；先做确定性 smoke/throughput，不读取留出结果后再调参。
- [ ] 为 `fixed`、`b128`、`2hash` 生成唯一运行清单；`2hash` 变体完成 hash 不相交和 SHA 校验。
- [ ] launcher 显式传 `--task sibling`，正式评价不自动生成新的 selection.tsv；结果复用检查任务、模型目录、评价器、探针和图身份。

本记录只冻结核对结果和阻断条件，不替代用户对两套上游配置的研究决策，也不构成 Sibling 已运行或有效性通过的声明。

## 5. 门禁实现追加记录（2026-10-16）

针对 §4 的任务/知识图门禁，新增了显式 `load_sibling_graph()` 接口，并将完整图目录及两
个文件 SHA 写入 `run_meta.json`/`rescore_meta.json`；`task=sibling` 缺少图目录时，
`train.py`、`rescore.py` 和 `run_stage0b.sh` 在启动前报错。用临时微型 fixture 完成了
接口 smoke，未使用官方数据、未生成正式权重，也未清除本记录中数据快照和步数门禁。

## 6. 公开 artifact 只读探针追加记录（2026-10-16）

通过 HF dataset API、文件树和 rows viewer 做了只读核验，并临时下载了两个 public parquet
以记录字节身份；原始 parquet 未提交到公开仓库。文件 SHA、字节数、字段、split 行数和
“缺少图/test/heldout”的边界见
[`outputs/sibling-confirmation-20261016/public-artifact-probe-20261016.json`](../outputs/sibling-confirmation-20261016/public-artifact-probe-20261016.json)。
该动作只固定了公开 data example 的输入/目标 artifact 身份，不把它升级为完整 Sibling
正式数据包；完整图仍须从获准的 notebook/再生源单独冻结并与数据建立对应记录。
