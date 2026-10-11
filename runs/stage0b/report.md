# 汇总（主指标 creativity；参照 ntp_fixed；margin 0.03；min_seeds 3）

| 标签 | 种子 | 选中步 | 开发集@选中 | 留出集@选中 | validity@选中 | memo_own@选中 | 终点 | 峰值 | 轨迹均值 | Δ vs 参照（均值 / 最小） | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ntp_fixed | 42,43,44 | 25000,20000,20000 | 0.208±0.014 (n=3) | 0.162±0.009 (n=3) | 0.211±0.008 (n=3) | 0.984±0.016 (n=3) | 0.161±0.011 (n=3) | 0.208±0.014 (n=3) | 0.124±0.008 (n=3) | — | — |
| ntp_fixed_grp | 42 | 35000 | 0.219 (n=1) | 0.167 (n=1) | 0.209 (n=1) | 1.000 (n=1) | 0.164 (n=1) | 0.219 (n=1) | 0.132 (n=1) | -0.012 / -0.012 (heldout) | INSUFFICIENT |
| setc_MM | 42 | 20000 | 0.180 (n=1) | 0.119 (n=1) | 0.150 (n=1) | 0.930 (n=1) | 0.023 (n=1) | 0.180 (n=1) | 0.075 (n=1) | -0.059 / -0.059 (heldout) | INSUFFICIENT |
| setc_MM_late_MN | 42 | 35000 | 0.250 (n=1) | 0.237 (n=1) | 0.312 (n=1) | 0.273 (n=1) | 0.227 (n=1) | 0.250 (n=1) | 0.227 (n=1) | +0.059 / +0.059 (heldout) | INSUFFICIENT |
| setc_MM_lock | 42 | 16000 | 0.148 (n=1) | 0.165 (n=1) | 0.213 (n=1) | 0.188 (n=1) | 0.148 (n=1) | 0.148 (n=1) | 0.145 (n=1) | -0.014 / -0.014 (heldout) | INSUFFICIENT |
| setc_MM_online | 42 | 15000 | 0.156 (n=1) | 0.176 (n=1) | 0.226 (n=1) | 0.047 (n=1) | 0.117 (n=1) | 0.156 (n=1) | 0.137 (n=1) | -0.002 / -0.002 (heldout) | INSUFFICIENT |
| setc_MN | 42,43,44 | 25000,35000,40000 | 0.240±0.005 (n=3) | 0.209±0.006 (n=3) | 0.273±0.012 (n=3) | 0.310±0.038 (n=3) | 0.193±0.025 (n=3) | 0.240±0.005 (n=3) | 0.150±0.012 (n=3) | +0.047 / +0.041 (heldout) | PASS |
| setc_NM | 42 | 20000 | 0.133 (n=1) | 0.115 (n=1) | 0.142 (n=1) | 0.797 (n=1) | 0.055 (n=1) | 0.133 (n=1) | 0.061 (n=1) | -0.063 / -0.063 (heldout) | INSUFFICIENT |
| tl_NN | 42 | 20000 | 0.156 (n=1) | 0.134 (n=1) | 0.169 (n=1) | 0.203 (n=1) | 0.031 (n=1) | 0.156 (n=1) | 0.060 (n=1) | -0.044 / -0.044 (heldout) | INSUFFICIENT |

† creativity 由旧 scores 的 n_unique_valid_novel / 测试探针数推得（事件 #4 修复前的运行仍受其影响）。
开发集峰值是对多个检查点取最大，系统性高估；判定只用留出集数值。

## 逐运行

| 运行 | 标签 | 种子 | 选中步 | 开发集@选中 | 留出集@选中 | 终点 | 峰值@步 |
|---|---|---|---|---|---|---|---|
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/ntp_fixed/s42 | ntp_fixed | 42 | 25000 | 0.227 | 0.178 | 0.180 | 0.227@25000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/ntp_fixed/s43 | ntp_fixed | 43 | 20000 | 0.180 | 0.147 | 0.141 | 0.180@20000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/ntp_fixed/s44 | ntp_fixed | 44 | 20000 | 0.219 | 0.162 | 0.164 | 0.219@20000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/ntp_fixed_grp/s42 | ntp_fixed_grp | 42 | 35000 | 0.219 | 0.167 | 0.164 | 0.219@35000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/setc_MM/s42 | setc_MM | 42 | 20000 | 0.180 | 0.119 | 0.023 | 0.180@20000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/setc_MM_late_MN/s42 | setc_MM_late_MN | 42 | 35000 | 0.250 | 0.237 | 0.227 | 0.250@35000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/setc_MM_lock/s42 | setc_MM_lock | 42 | 16000 | 0.148 | 0.165 | 0.148 | 0.148@16000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/setc_MM_online/s42 | setc_MM_online | 42 | 15000 | 0.156 | 0.176 | 0.117 | 0.156@15000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/setc_MN/s42 | setc_MN | 42 | 25000 | 0.250 | 0.219 | 0.195 | 0.250@25000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/setc_MN/s43 | setc_MN | 43 | 35000 | 0.234 | 0.197 | 0.148 | 0.234@35000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/setc_MN/s44 | setc_MN | 44 | 40000 | 0.234 | 0.211 | 0.234 | 0.234@40000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/setc_NM/s42 | setc_NM | 42 | 20000 | 0.133 | 0.115 | 0.055 | 0.133@20000 |
| /home/wlwuser/P6-Roll_the_dice/runs/stage0b/tl_NN/s42 | tl_NN | 42 | 20000 | 0.156 | 0.134 | 0.031 | 0.156@20000 |
