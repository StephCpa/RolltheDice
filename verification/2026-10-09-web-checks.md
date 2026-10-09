# P6 核验记录：原论文、代码与近邻工作（2026-10-09）

按仓库约定（AGENTS.md 第 5 条）登记：来源版本、定位、核验动作、日期、范围与未覆盖部分。本记录只登记"来源说了什么"，科学判断见 `analysis/`。

- **核验日期**：2026-10-09
- **核验动作**：WebFetch/WebSearch 在线访问（本人直接执行）＋ 两个并行查新子代理（各返回逐篇核验日志）；GitHub API 读取仓库元数据与脚本内容。
- **对照对象**：`sources/proposal-20261004.md`（ChatGPT 方案，自署查新截止 2026-10-04）。
- **未覆盖**：未下载 PDF 逐页比对；未运行任何代码或模型；未核验 arXiv 页面以外的会议论文集正文（除 ACL/PMLR/ICR 落地页）；未复核子代理引用的每一条补充搜索结果原文。

---

## A. 原论文与代码

### A1. 论文本体（arXiv:2504.15266）

| 项目 | 核验结果 | 方案主张 |
|---|---|---|
| 标题/作者 | Roll the dice & look before you leap: Going beyond the creative limits of next-token prediction；Vaishnavh Nagarajan, Chen Henry Wu, Charles Ding, Aditi Raghunathan | 一致 |
| 版本史 | v1 2025-04-21，v2 2025-05-26，v3 2025-07-14，**v4 2025-08-28（最新）** | 方案称核对 v4（2025 年 8 月修订）——一致 |
| 奖项 | **ICML 2025 Outstanding Paper Award（官方奖项页 6 篇之一，已确认）**；arXiv comments 标 "ICML 2025 (oral)" | 方案称 Outstanding Paper——一致 |
| 任务 | Sibling Discovery、Triangle Discovery、Circle Construction、Line Construction | 一致（四类） |
| teacherless 训练 | 仅给定 prompt 与 dummy 位置，一次性并行预测全部 token；实际与 NTP 目标混合使用 | 与方案表述一致 |
| 模型 | Gemma v1 2B（主实验）、GPT-2 86M（非嵌入参数）、SEDD 90M | 方案复现范围判断依据成立 |

**B.1 相关（部分核验）**：HTML 版正文可检索到关键句 "the seeds we provide are arbitrarily paired with the training datapoint" 与 "we do not understand this presently"；正文同时承认不清楚该想法对真实数据的泛化。**但 B.1 节（"Style of noise-injection"）正文在 HTML 抓取中被截断，未能引用 B.1 原句**；方案所述"传统生成模型要求噪声分布与数据分布整体匹配"的对比表述未在可获取部分逐字出现（正文最接近的是 §5.1 关于温度采样边缘化 vs 固定种子"每次落实一个想法"的直觉讨论）。**判定：方案引用的开放问题本身成立（有正文原句支撑），B.1 的具体措辞待 PDF 逐页核对。**

### A2. 代码仓库（github.com/ChenWu98/algorithmic-creativity）

| 项目 | 核验结果 | 方案主张 |
|---|---|---|
| HEAD 快照 | `7176dc7d35ca17605a76fa873ec97ad62a8537ac`（2025-05-26, "Update README.md"，共 11 次提交） | 方案称快照 `7176dc7`——一致 |
| 硬件 | README："All experiments can be run on a single A6000 GPU. Batch sizes are tuned on this device." | 一致 |
| 模型范围 | README 明示：论文含 Gemma 2B 与 GPT-2/SEDD，**仓库只含 GPT-2/SEDD 代码**（NTP / teacherless / SEDD 三路线，四任务目录） | 一致 |
| Circle NTP 启动脚本 | `run_train.sh` → `train.sh circle.10.9.10.10000 0.0 12 0`；`train.sh` 实测：`MODEL_PATH=gpt2`、`max_seq_length/max_length/block_size 128`、`train_batch_size 64`、`max_steps 40000`、`--fp16`、`DATASET=/data/locus/project_data/.../creativity_data/`（本地路径需改） | 全部逐项一致 |
| 注意（本核验新增） | **HEAD 提交（2025-05-26）早于论文 v3（07-14）与 v4（08-28）**：公开代码对应约 v2 时期，与 v4 正文可能存在细节差异，复现时不应当作 v4 的精确实现 | 方案未提及 |

## B. 近邻工作逐篇核验（两个子代理日志合并）

标记：✔=方案描述准确；△=大体准确有小偏差；✖=未核验到。"—"=与集合级训练配对无关。

| # | 工作 | 存在性/状态 | 作者与原论文重叠 | 方案描述准确性 | 是否"训练阶段 seed 集合↔解集合配对" |
|---|---|---|---|---|---|
| 1 | Gacha Decoding（arXiv:2610.01382） | ✔ 2026-10-01 v1 预印本（Geng, Zhang, Lee, Li, Ghazvininejad, Koh） | 无 | ✔（推理时"指令遵循+外部 RNG"）；注：纯 inference-time | 否 |
| 2 | Intent Factored Generation（arXiv:2506.09659） | ✔ 2025-06-11 v1 预印本（Ahmed 等，含 J.N. Foerster） | 无 | ✔（高温意图→低温回答，免训练） | 否 |
| 3 | String Seed of Thought（ICLR 2026 会议录） | ✔ ICLR 2026 正式发表（Misaki, Akiba） | 无 | ✔ | 否（纯提示） |
| 4 | Where You Inject Diversity Matters（arXiv:2606.10302） | ✔ 2026-06-09 v1 预印本（Zhang, Xin, Zhong） | 无 | ✔（test-time 多样性注入位置框架＋规范级生成） | 否 |
| 5 | LaDiR（arXiv:2510.04573） | ✔ v1 2025-10-06，v6 2026-04-23；**ICLR 2026 Poster**（OpenReview 确认） | 无 | △（潜扩散规划＋自回归解码正确；arXiv 页不显示录用信息） | 有训练但**单点噪声→单点潜轨迹**流匹配，非集合级 |
| 6 | Mode-Conditioning（arXiv:2512.01127） | ✔ 2025-11-30 v1 预印本 | **Chen Henry Wu＋Aditi Raghunathan**（方案所称团队延续成立） | ✔（含梯度聚类自动找模式） | 否（推理时预算分配＋配套条件训练） |
| 7 | Prefix-Conditioned SFT（ACL 2026 long.9） | ✔ ACL 2026 长文，实际标题 *Learning Diverse Responses with Prefix-Conditioned Supervised Fine-Tuning*（Fan 等） | 无 | △：前缀"分离多答案训练条件"✔；**"随机前缀"与"每轮重采样前缀"两细节摘要未提及、PDF 未能读取——未核验** | 否（确定性条件→单点答案） |
| 8 | Annotations Mitigate Post-Training Mode Collapse（PMLR v306） | ✔ ICML 2026 正式发表（Springer 等 11 人） | **Aditi Raghunathan** | △（"保存注释分布"✔；"语义分布与条件回答分开"系转述，非原文框架） | 否 |
| 9 | IMLE（arXiv:1809.09087，NeurIPS 2018） | ✔（Li, Malik） | — | ✔ | **最接近**：每个噪声样本在目标集合中动态选最近邻（非对称、可撞车），**非全局双射最优配对** |
| 10 | GFlowNet-LLM 摊销推断（arXiv:2310.04363，ICLR 2024） | ✔（Hu, Jain 等，含 Bengio） | — | ✔ | 否（分布级匹配，无当轮排列分配步骤） |
| 11 | ExplainFuzz（arXiv:2604.06559） | ✔ 2026-04-08 预印本（Baiget 等，含 Van den Broeck） | — | ✔（概率电路约束条件化测试生成） | 否 |

## C. 补充查新（方案之外的新发现）

两个子代理各自独立补充检索（2026-10-09），要点：

1. **未发现** 2026-10-09 前公开的、与"训练阶段一组种子共同覆盖一组已验证答案（集合级配对）"直接相同的先行工作。
2. 方案**遗漏的概念近邻**（后续 related-work 必须纳入）：
   - **OTSeq2Seq**（*Improving Sequence-to-Sequence Learning via Optimal Transport*）：模型采样句 vs 真值句的最优传输软匹配训练——**思想上最接近"生成样本与目标集合匹配训练"的先例**（RNN seq2seq 时代，OT 软分配而非严格排列）。
   - **OTSeq2Set**（EMNLP 2022）：OT 增强的 sequence-to-set，含一对一/多对一匹配——**与"多个有效答案的集合匹配"最相关**。
   - *Sequence Generation with OT-Enhanced RL*（AAAI）：跨位置词级动态匹配。
   - Golden Noise for Diffusion Models（NeurIPS 2025）：逐 seed 噪声优化（图像域、逐点）。
   - InternBootcamp（arXiv:2508.08636）：大规模可验证多解任务数据（多解监督，无 seed 配对）。
3. **未发现** IMLE 直接用于 LLM 多参考答案微调的公开工作；IMLE 谱系集中在视觉/机器人（IMLE Policy RSS 2025、Adaptive IMLE ICML 2023 等）。

## D. 对方案的核验结论

- 方案的**事实层**（原论文贡献、奖项、版本、代码范围、启动脚本参数、13 篇近邻的存在与定性）**全部核验通过或仅有措辞级偏差**；两处 ✖/△（B.1 逐字、P-SFT 重采样细节）已在上面标明，不影响主线。
- 方案自我定位"候选构造，不是已确认新颖的方法"与其查新结果一致；本核验进一步把新颖性边界**收窄到**：相对 IMLE（非对称最近邻）、OTSeq2Seq/OTSeq2Set（RNN 时代 OT 匹配）、P-SFT（冻结条件分配）而言，候选的差异点是 **LLM/decoder-only 设置＋已验证解集合＋teacherless 接口带来的 m 次前向计算 m² 个代价＋迁移/组合泛化评价**。该差异是否构成实质增量，属 `analysis/` 的判断范围。
