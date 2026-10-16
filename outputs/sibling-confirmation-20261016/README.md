# Sibling 三臂确认准备包（development）

本目录记录事件 #9 修正后、进入 Sibling 前的同系统确认方案。它是**启动门禁和预注册草案**，不是实验结果；本次没有启动 Sibling 训练、没有读取服务器权重，也没有进行留出评价。

机器可读的唯一入口是 [`freeze_manifest.json`](freeze_manifest.json)。它列出固定三臂、三个训练种子、候选训练预算、数据与知识图要求、评价身份字段和停止规则。状态为 `blocked_pending_artifacts`，表示在数据快照、完整知识图、变体校验和 smoke/吞吐核价完成前不得启动九个正式运行。

本轮采用 P6 当前 trainer/model 栈做同系统确认：`gpt2`、序列长度 128、学习率 `1e-4`、warmup 2000；计划步数暂按 Sibling NTP 的 800k 预算登记，仍须经过无留出 smoke/吞吐核价。这个 profile 不宣称复现上游 8-layer fresh-init 训练。

三臂是 `ntp_fixed`（64 序列/步）、`ntp_fixed_b128`（128 序列/步）和 `ntp_fixed_2hash`（`dup_hash(k=2)`、128 序列/步），种子为 42、43、44。主比较是 `2hash − b128`，同时报告 `2hash − fixed`。不加入 `2tgt`、sample、k 扫描、Hungarian、teacherless、新正则或 scale。

启动前必须补齐：官方 `sibling.5.500.10.50000` 的本地快照及 SHA、`entities_b1_dict.json`/`entities_b2_dict.json` 完整知识图、2hash 变体的不相交性和 SHA、任务显式传递，以及无留出确定性 smoke/吞吐记录。任何 smoke 输出只能作为工程门禁证据，不能写成 Sibling 有效性证据。

本轮只读探针确认了 HF public artifact 的 revision、train/valid parquet SHA 和字段；探针记录见 [`public-artifact-probe-20261016.json`](public-artifact-probe-20261016.json)。该公开 data example 没有完整知识图或 test/heldout 文件，因此不能直接作为正式评价输入。
