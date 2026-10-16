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

## 当前状态

数据生成命令已在远程成功完成。随后远程 SSH 出现连接/握手超时，尚未启动训练、未生成
checkpoint、未生成 heldout 预测，也没有改变任何已有运行。
