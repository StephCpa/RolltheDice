# 失效运行登记
tl_hybrid 首轮（2026-10-09）：训练器把 hybrid JSON（本身已 1:3 展开）再按普通数据展开，
56% 单元变成'教 eos-run'，模型答案槽 EOS 占优，全指标 0。根因已修（train.py tl_hybrid 分支
按 target_labels 逐条消费）。本目录保留作证据，不进入任何结论。修正版见 tl_hybrid_v2。
