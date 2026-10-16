# ICML 2027 执行状态与证据门槛

日期：2026-10-17  
状态：`development` 计划记录；不含正式 Sibling 结果。

## 当前论文可行性判断

本工作可以形成一篇较完整的 ICML 论文，但当前证据仍不足以支撑提交。论文的可交付核心应是：

> 固定多入口配对如何改变 seed-conditioned 生成的训练行为，以及这种收益在哪些任务、模型规模和曝光控制下成立。

现有 Circle 结果提供了动机和候选配方；修复后的 Line 结果结束了 `ntp_2tgt` 的主检验，不能再作为继续追门槛的理由。是否能冲 ICML，取决于下面的独立确认门槛，而不是继续扩大旧候选网格。

## 已完成与正在进行

### 已完成

1. 事件 #9 的 Line 评价器任务身份、checkpoint 身份和去重口径已修复；`ntp_2tgt` 主检验结束，`2hash` 保留为下一轮候选。
2. 负责人接受 `derived-5x500`，登记为本项目新的 Sibling 确认协议。
3. 生成并冻结一次 fresh heldout：2048 行，seed `2026101701`，SHA256
   `edcdd3baa2dbe0067f0cb3a24cd60a7a7c84f3203bd640d625111ad42b95686a`。
4. Sibling 三臂清单已冻结：`ntp_fixed`、`ntp_fixed_b128`、`ntp_fixed_2hash`，每臂 seed 42/43/44；主比较为 `2hash-b128`，次比较为 `2hash-fixed`。
5. Circle development 的三臂 fresh 重评分已完成并另行记录；它是单 seed 的开发证据，不计入 Sibling 独立确认。

### 正在进行

首个正式 Sibling 运行 `ntp_fixed_2hash/seed42` 已启动，协议为 800,000 steps、task `sibling`、冻结数据与图。尚未产生正式 heldout 结果，不能据此报告科学结论。其余八个运行等待安全的 GPU 资源；共享 GPU 上存在其他用户残留占用，不能强行启动。

## ICML 必须门槛

### Gate A：Sibling 独立确认

完成九个运行后，按冻结的 development checkpoint selection 选点，再对同一 fresh heldout 做一次性评价。报告逐 seed 差值、均值和区间，不只报告 PASS/FAIL。

建议沿用项目数值门槛：主比较 `2hash-b128` 和次比较 `2hash-fixed` 均要求平均差至少 0.03 且三个 seed 方向一致；如果任务尺度不支持该幅度，必须在查看结果前重新登记替代门槛，不能事后调节。

Gate A 通过后，才可以把“固定多入口在两个任务上可复现”作为主结果。若只在一个任务通过，论文必须把适用边界写进主结论。

### Gate B：最小曝光控制

在 Gate A 之后执行 `repeat_rows_k2`：保持 hash→answer 边和唯一边数不变，每个训练行重复两次，使训练序列数与 `2hash` 对齐。该对照只检验重复曝光能否解释收益，不扩展候选网格。

### Gate C：规模点

只有 Gate A 方向成立后，运行 GPT-2 medium；优先 Circle 与 Sibling 的 `fixed`/`2hash` 三 seed。2B 是 stretch，不是 ICML 主线的启动条件。若规模点失败，应如实限定模型规模，而不是把小模型结果外推成普遍规律。

## 理论最低交付

理论部分不需要声称“监督图越密必然更好”。最低充分版本是一个带明确假设的受限共享表示命题：在答案相关表示可由多个固定 hash 入口共享、而重复同一 hash 不增加入口覆盖的条件下，`2hash` 可以降低未见 hash 的预测风险；随后给出 one-hot 或无共享表示反例，说明结论不是普适定理。

理论预测必须由 Gate B 的 `repeat_rows_k2` 与入口覆盖读数检验。若曝光控制不支持预测，理论应降级为机制假设，不能包装成已证实定理。

## 当前禁止事项

- 不再追加 `ntp_2tgt` Line 训练来追旧门槛；
- 不启动 Hungarian、teacherless、新正则项、更多 k 扫描或 Triangle 网格；
- 不把 Circle 单 seed development 重评分、工程 smoke 或已看过的 Line heldout 当作独立确认；
- 不在共享 GPU 上为赶时间抢占其他用户资源；
- 不在正式 Sibling 结果出现前写 PASS、泛化或规模结论。

