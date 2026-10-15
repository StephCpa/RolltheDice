# 汇总（主指标 creativity；参照 ntp_fixed；margin 0.03；min_seeds 3）

| 标签 | 种子 | 选中步 | 开发集@选中 | 留出集@选中 | validity@选中 | memo_own@选中 | 终点 | 峰值 | 轨迹均值 | Δ vs 参照（均值 / 最小） | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ntp_2tgt | 42,43,44 | 20000,25000,35000 | 0.346±0.007 (n=3) | 0.294±0.016 (n=3) | 0.407±0.014 (n=3) | 0.479±0.018 (n=3) | 0.286±0.037 (n=3) | 0.346±0.007 (n=3) | 0.246±0.010 (n=3) | +0.132 / +0.101 (heldout) | PASS |
| ntp_2tgt_sample | 42,43,44 | 35000,40000,25000 | 0.286±0.052 (n=3) | 0.228±0.018 (n=3) | 0.303±0.030 (n=3) | 0.422±0.036 (n=3) | 0.216±0.030 (n=3) | 0.286±0.052 (n=3) | 0.134±0.020 (n=3) | +0.066 / +0.014 (heldout) | PASS |
| ntp_fixed | 42,43,44 | 25000,20000,20000 | 0.208±0.014 (n=3) | 0.162±0.009 (n=3) | 0.211±0.008 (n=3) | 0.984±0.016 (n=3) | 0.161±0.011 (n=3) | 0.208±0.014 (n=3) | 0.124±0.008 (n=3) | — | — |
| ntp_fixed_2hash | 42,43,44 | 40000,15000,15000 | 0.263±0.007 (n=3) | 0.241±0.009 (n=3) | 0.319±0.008 (n=3) | 0.841±0.095 (n=3) | 0.224±0.032 (n=3) | 0.263±0.007 (n=3) | 0.201±0.007 (n=3) | +0.078 / +0.057 (heldout) | PASS |
| ntp_fixed_b128 | 42,43,44 | 35000,10000,25000 | 0.193±0.021 (n=3) | 0.165±0.005 (n=3) | 0.211±0.006 (n=3) | 0.924±0.076 (n=3) | 0.167±0.017 (n=3) | 0.193±0.021 (n=3) | 0.152±0.013 (n=3) | +0.003 / -0.023 (heldout) | FAIL |
| ntp_k2c | 42,43,44 | 25000,20000,40000 | 0.286±0.021 (n=3) | 0.244±0.010 (n=3) | 0.392±0.043 (n=3) | 0.484±0.012 (n=3) | 0.227±0.024 (n=3) | 0.286±0.021 (n=3) | 0.202±0.014 (n=3) | +0.081 / +0.052 (heldout) | PASS |
| ntp_k3c | 42,43,44 | 15000,15000,15000 | 0.180±0.023 (n=3) | 0.200±0.018 (n=3) | 0.416±0.058 (n=3) | 0.016±0.012 (n=3) | 0.039±0.016 (n=3) | 0.180±0.023 (n=3) | 0.078±0.006 (n=3) | +0.038 / -0.014 (heldout) | FAIL |
| ntp_k4c | 42,43,44 | 15000,15000,15000 | 0.174±0.027 (n=3) | 0.176±0.040 (n=3) | 0.498±0.078 (n=3) | 0.023±0.023 (n=3) | 0.036±0.007 (n=3) | 0.174±0.027 (n=3) | 0.054±0.012 (n=3) | +0.014 / -0.037 (heldout) | FAIL |

† creativity 由旧 scores 的 n_unique_valid_novel / 测试探针数推得（事件 #4 修复前的运行仍受其影响）。
开发集峰值是对多个检查点取最大，系统性高估；判定只用留出集数值。

## 逐运行

| 运行 | 标签 | 种子 | 选中步 | 开发集@选中 | 留出集@选中 | 终点 | 峰值@步 |
|---|---|---|---|---|---|---|---|
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_2tgt/s42 | ntp_2tgt | 42 | 20000 | 0.344 | 0.314 | 0.242 | 0.344@20000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_2tgt/s43 | ntp_2tgt | 43 | 25000 | 0.359 | 0.305 | 0.359 | 0.359@25000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_2tgt/s44 | ntp_2tgt | 44 | 35000 | 0.336 | 0.263 | 0.258 | 0.336@35000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_2tgt_sample/s42 | ntp_2tgt_sample | 42 | 35000 | 0.203 | 0.192 | 0.172 | 0.203@35000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_2tgt_sample/s43 | ntp_2tgt_sample | 43 | 40000 | 0.273 | 0.248 | 0.273 | 0.273@40000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_2tgt_sample/s44 | ntp_2tgt_sample | 44 | 25000 | 0.383 | 0.245 | 0.203 | 0.383@25000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_fixed/s42 | ntp_fixed | 42 | 25000 | 0.227 | 0.178 | 0.180 | 0.227@25000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_fixed/s43 | ntp_fixed | 43 | 20000 | 0.180 | 0.147 | 0.141 | 0.180@20000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_fixed/s44 | ntp_fixed | 44 | 20000 | 0.219 | 0.162 | 0.164 | 0.219@20000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_fixed_2hash/s42 | ntp_fixed_2hash | 42 | 40000 | 0.273 | 0.235 | 0.273 | 0.273@40000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_fixed_2hash/s43 | ntp_fixed_2hash | 43 | 15000 | 0.266 | 0.229 | 0.234 | 0.266@15000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_fixed_2hash/s44 | ntp_fixed_2hash | 44 | 15000 | 0.250 | 0.258 | 0.164 | 0.250@15000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_fixed_b128/s42 | ntp_fixed_b128 | 42 | 35000 | 0.180 | 0.155 | 0.180 | 0.180@35000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_fixed_b128/s43 | ntp_fixed_b128 | 43 | 10000 | 0.234 | 0.173 | 0.188 | 0.234@10000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_fixed_b128/s44 | ntp_fixed_b128 | 44 | 25000 | 0.164 | 0.167 | 0.133 | 0.164@25000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_k2c/s42 | ntp_k2c | 42 | 25000 | 0.258 | 0.230 | 0.195 | 0.258@25000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_k2c/s43 | ntp_k2c | 43 | 20000 | 0.328 | 0.263 | 0.211 | 0.328@20000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_k2c/s44 | ntp_k2c | 44 | 40000 | 0.273 | 0.237 | 0.273 | 0.273@40000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_k3c/s42 | ntp_k3c | 42 | 15000 | 0.203 | 0.165 | 0.008 | 0.203@15000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_k3c/s43 | ntp_k3c | 43 | 15000 | 0.133 | 0.218 | 0.062 | 0.133@15000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_k3c/s44 | ntp_k3c | 44 | 15000 | 0.203 | 0.217 | 0.047 | 0.203@15000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_k4c/s42 | ntp_k4c | 42 | 15000 | 0.219 | 0.256 | 0.047 | 0.219@15000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_k4c/s43 | ntp_k4c | 43 | 15000 | 0.125 | 0.147 | 0.039 | 0.125@15000 |
| /home/lzn/P6-RolltheDice/runs/stage0b/ntp_k4c/s44 | ntp_k4c | 44 | 15000 | 0.180 | 0.125 | 0.023 | 0.180@15000 |
