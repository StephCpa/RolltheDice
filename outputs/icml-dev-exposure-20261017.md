# Circle exposure-control staging — 2026-10-17

状态：development staging；没有训练输出或科学结论。

## 远程数据变体

服务器：`gpu-gfkd`  
基础数据：`/home/wlwuser/P6-Roll_the_dice/data/circle.10.9.10.10000`  
基础训练文件 SHA256：`5474588dedca883d46c528be0d2eda1eb0752e45e708e55e0be9bfa26c71af80`

已生成两个只读派生目录：

| 变体 | 训练行数 | unique hash | unique edge | train SHA256 |
|---|---:|---:|---:|---|
| `circle.10.9.10.10000.repeat2.20261017` | 20,000 | 10,000 | 10,000 | `e2a0d0602bde7bc4a640a839d608cf8ea610b0d1fa3b261e265fcbce3c3831d5` |
| `circle.10.9.10.10000.dup2.20261017` | 20,000 | 20,000 | 20,000 | `0bf18c1dea39c321df4d20a1130c09c355448c7e179498ff7660dab10ccfa777` |

`repeat2` 保持原 hash→answer 图，仅重复每条训练行两次；`dup2` 为每个答案增加一个
新 hash，保持每个 hash 一个目标。两个变体的 valid/test/heldout 文件由基础目录复制。

## 预定比较

使用同一 Circle 配置、同一 checkpoint 选择和 fresh heldout：

`ntp_fixed`、`ntp_fixed_b128`、`ntp_fixed_2hash`、`ntp_fixed_repeat2`。

`2hash` 与 `repeat2` 的差异用于区分“新增 hash 入口”与“相同边重复曝光”。此 staging
比较尚未预注册为正式结果；在 GPU 调度安全且补充协议签署前，不得报告 PASS/FAIL。

## 工程 smoke

在 GPU5 上用缓存的远程 GPT-2 checkpoint（`runs/stage0b/ntp_fixed/s42/final_model`）完成了
`repeat2` 的 2-step、batch 4、4 probes smoke。`run_meta.json`、`scores.jsonl`、
`throughput.json` 和 480 MB 的 `final_model` 均生成；训练器、tokenizer、CUDA、数据变体和
保存路径均可工作。该输出位于远程 `runs/icml2027-dev-20261017/smoke/trace_s42/`，
只验证工程链路；2-step 的全零 test creativity 不作科学解释，也不计入论文结果。

## 当前状态

数据生成命令已在远程成功完成。随后曾短暂启动三条 `s42` 工程试跑：`ntp_fixed`
batch 64、`ntp_fixed` batch 128，以及 `dup2` batch 128。它们误用了已训练的
`stage0b/ntp_fixed/s42/final_model`，分别只推进到约 500--1500 steps；审计发现服务器
已有完整离线 `gpt2` 快照
`/home/wlwuser/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721d8c4cfcd7e`，
故三条试跑已由操作者终止，输出不得评分或复用，只保留为吞吐和显存工程记录。

下一次启动必须让全部臂使用同一份不可变的离线 `gpt2` 快照，并在 `run_meta.json` 中记录
模型配置与权重身份。当前没有完整曝光控制训练或 heldout 预测，所有新增运行仍属于
development。

## corrected development launch

At 04:28--04:30 CST, the following three development runs were relaunched from the same
offline snapshot (no heldout score is available yet):

| run | data | batch | GPU | remote output |
|---|---|---:|---:|---|
| fixed s42 | base | 64 | 5 | `exposure-gpt2/ntp_fixed/s42` |
| fixed-b128 s42 | base | 128 | 6 | `exposure-gpt2/ntp_fixed_b128/s42` |
| 2hash s42 | `dup2` | 128 | 7 | `exposure-gpt2/ntp_fixed_2hash/s42` |

The snapshot is `gpt2-local-snapshot-607a30d7`; its `model.safetensors` SHA256 is
`248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`. Each run uses
40,000 steps, warmup 2,000, seed 42, evaluation/checkpoint interval 5,000, and 128 probes.
The three runs are still engineering-stage development runs. Their outputs require a fresh
selection and heldout protocol before any scientific comparison.

## first checkpoint gate

At 04:50 CST the three `s42` runs were verified alive on GPUs 5--7. Each has a complete
`ckpts/step5000/` directory, `preds_step5000.jsonl`, `scores.jsonl`, and `run_meta.json`.
The three `run_meta.json` files point to the same offline GPT-2 snapshot and record the same
configuration/tokenizer identities. The built-in step-5000 scores are engineering diagnostics
only: they use the training launch probe settings and are not the fresh heldout endpoint.
They currently contain zero valid generated answers at this early checkpoint, so no effect
claim is made from them.

## checkpoint I/O observation

During the 15k checkpoint, the three processes shared the `/home/wlwuser` ext4 volume and
entered `balance_dirty_pages`/filesystem waits while writing their approximately 480 MB model
files. The `2hash` checkpoint completed first and its process resumed training at step 16,500;
the fixed and b128 processes were still writing their 15k weights at the last check. This is
an infrastructure throughput observation only. No process was interrupted and no partial
checkpoint is eligible for scoring.
