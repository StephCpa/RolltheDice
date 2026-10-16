# Sibling formal launch record: ntp_fixed_2hash / seed 42 (development ledger)

日期：2026-10-17（Asia/Shanghai）  
服务器：`gpu-gfkd`  
冻结输入：`freeze_manifest_ready-derived-5x500-20261017.json`；数据副本
`data/sibling.accepted-derived-5x500.20261017`；模型 revision
`gpt2-local-snapshot-607a30d7`；P6 code SHA `0f8cee6`。

## 启动状态

在 Circle 暴露控制批次的 fixed 与 2hash 已完成、GPU7 空闲后，使用
`run_sibling_confirmation.py --start` 启动：

- arm：`ntp_fixed_2hash`
- seed：42
- batch：128
- steps：800,000
- task：`sibling`
- graph：显式 `data/sibling.accepted-derived-5x500.20261017/graph`
- 输出：`runs/icml2027-sibling-confirmation-20261017/ntp_fixed_2hash__seed42`
- 日志：`runs/icml2027-sibling-confirmation-20261017/logs/ntp_fixed_2hash__seed42.log`
- CUDA：7
- 启动时间（run_meta）：2026-10-17 06:50:00

子进程已确认存活并写出 `run_meta.json`；约 1,700 steps 时日志仍在推进，
`git_commit=0f8cee6`、图 schema 和两份图 SHA 均已写入。当前没有 checkpoint
或 heldout 结果，不能作科学解释。

## 口径

这是冻结三臂×三种子确认包的第一条已启动运行。其余八条必须使用同一冻结输入，
按 GPU 空闲情况逐条启动；不得追加臂、改变步数、改写 heldout 或边跑边选对照。
