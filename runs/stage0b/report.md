# 汇总（主指标 creativity；参照 ntp_fixed；margin 0.03；min_seeds 3）

| 标签 | 种子 | 选中步 | 开发集@选中 | 留出集@选中 | validity@选中 | memo_own@选中 | 终点 | 峰值 | 轨迹均值 | Δ vs 参照（均值 / 最小） | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s42 | 42,42,42,42,42,42,42,42,42,42,42,42,42 | 20000,20000,25000,35000,20000,35000,16000,15000,25000,20000,20000,30000,35000 | 0.216±0.017 (n=13) | 0.185±0.015 (n=13) | 0.239±0.021 (n=13) | 0.462±0.101 (n=13) | 0.150±0.020 (n=13) | 0.216±0.017 (n=13) | 0.133±0.016 (n=13) | — | — |
| s43 | 43,43,43,43 | 25000,20000,35000,30000 | 0.275±0.042 (n=4) | 0.244±0.043 (n=4) | 0.325±0.058 (n=4) | 0.428±0.198 (n=4) | 0.229±0.052 (n=4) | 0.275±0.042 (n=4) | 0.181±0.033 (n=4) | — | — |
| s44 | 44,44,44,44 | 35000,20000,40000,35000 | 0.260±0.026 (n=4) | 0.207±0.021 (n=4) | 0.279±0.036 (n=4) | 0.471±0.207 (n=4) | 0.217±0.020 (n=4) | 0.260±0.026 (n=4) | 0.151±0.026 (n=4) | — | — |

† creativity 由旧 scores 的 n_unique_valid_novel / 测试探针数推得（事件 #4 修复前的运行仍受其影响）。
开发集峰值是对多个检查点取最大，系统性高估；判定只用留出集数值。

## 逐运行

| 运行 | 标签 | 种子 | 选中步 | 开发集@选中 | 留出集@选中 | 终点 | 峰值@步 |
|---|---|---|---|---|---|---|---|
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/ntp_2tgt/s42 | s42 | 42 | 20000 | 0.344 | 0.314 | 0.242 | 0.344@20000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/ntp_2tgt/s43 | s43 | 43 | 25000 | 0.359 | 0.305 | 0.359 | 0.359@25000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/ntp_2tgt/s44 | s44 | 44 | 35000 | 0.336 | 0.263 | 0.258 | 0.336@35000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/ntp_2tgt_unif/s42 | s42 | 42 | 20000 | 0.305 | 0.193 | 0.148 | 0.305@20000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/ntp_fixed/s42 | s42 | 42 | 25000 | 0.227 | 0.178 | 0.180 | 0.227@25000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/ntp_fixed/s43 | s43 | 43 | 20000 | 0.180 | 0.147 | 0.141 | 0.180@20000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/ntp_fixed/s44 | s44 | 44 | 20000 | 0.219 | 0.162 | 0.164 | 0.219@20000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/ntp_fixed_grp/s42 | s42 | 42 | 35000 | 0.219 | 0.167 | 0.164 | 0.219@35000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/setc_MM/s42 | s42 | 42 | 20000 | 0.180 | 0.119 | 0.023 | 0.180@20000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/setc_MM_late_MN/s42 | s42 | 42 | 35000 | 0.250 | 0.237 | 0.227 | 0.250@35000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/setc_MM_lock/s42 | s42 | 42 | 16000 | 0.148 | 0.165 | 0.148 | 0.148@16000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/setc_MM_online/s42 | s42 | 42 | 15000 | 0.156 | 0.176 | 0.117 | 0.156@15000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/setc_MN/s42 | s42 | 42 | 25000 | 0.250 | 0.219 | 0.195 | 0.250@25000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/setc_MN/s43 | s43 | 43 | 35000 | 0.234 | 0.197 | 0.148 | 0.234@35000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/setc_MN/s44 | s44 | 44 | 40000 | 0.234 | 0.211 | 0.234 | 0.234@40000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/setc_NM/s42 | s42 | 42 | 20000 | 0.133 | 0.115 | 0.055 | 0.133@20000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/tl_NN/s42 | s42 | 42 | 20000 | 0.156 | 0.134 | 0.031 | 0.156@20000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/tl_RN/s42 | s42 | 42 | 30000 | 0.195 | 0.179 | 0.188 | 0.195@30000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/tl_RN/s43 | s43 | 43 | 30000 | 0.328 | 0.328 | 0.266 | 0.328@30000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/tl_RN/s44 | s44 | 44 | 35000 | 0.250 | 0.191 | 0.211 | 0.250@35000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/tl_RN_unif/s42 | s42 | 42 | 35000 | 0.242 | 0.204 | 0.227 | 0.242@35000 |
