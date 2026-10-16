# Sibling 正式运行启动记录：fixed / b128 seed 42

日期：2026-10-17  
状态：正式训练已启动；尚无 heldout 科学结果。

## 启动的两行

两行均由 `staging-0f8cee6/code/run_sibling_confirmation.py --start` 启动，读取同一份
`ready_for_formal_run` freeze manifest 和 run manifest：

| arm | seed | GPU | 输出目录 | 远端 train PID | 启动证据 |
|---|---:|---:|---|---:|---|
| `ntp_fixed` | 42 | 4 | `runs/icml2027-sibling-confirmation-20261017/ntp_fixed__seed42` | 4038863 | 目录已创建，`train.py` 正在运行 |
| `ntp_fixed_b128` | 42 | 5 | `runs/icml2027-sibling-confirmation-20261017/ntp_fixed_b128__seed42` | 4039301 | 目录已创建，`train.py` 正在运行 |

两行的共同身份：

- task：`sibling`；
- planned steps：800,000；`eval_every=20,000`；`ckpt_every=20,000`；
- model：本地 GPT-2 snapshot `607a30d783dfa663caf39e06633721c8d4cfcd7e`；
- model revision：`gpt2-local-snapshot-607a30d7`；
- accepted data：`sibling.accepted-derived-5x500.20261017`；
- graph：同一冻结 Sibling graph；
- batch：`ntp_fixed=64`，`ntp_fixed_b128=128`。

## 资源与证据边界

启动前 GPU4/GPU5 分别报告约 22.6 GB/22.5 GB free；两卡均有其他用户进程，故采用
一臂一卡并保留原进程，不做抢占。启动后远端进程表确认两个 `train.py` 子进程存在；
模型加载期间显存分配尚未稳定，后续需以新的 `nvidia-smi` 和 checkpoint 文件核对进度。

这份记录只证明正式运行已按冻结身份启动，不证明训练完成、checkpoint 可用或任何科学
结论。fresh heldout 仍只能在全部正式运行完成后按冻结选择协议执行。

