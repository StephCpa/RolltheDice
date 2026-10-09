# 失效运行登记（事件 #3，2026-10-10）
train.py group_step 的 NTP 前向误取 groups_ex[g][j] 的整个元素（前缀+答案均来自 j），
导致本臂每步训练的是名义配对——随机双射从未生效。本臂数据等价于
"组结构批组织的 ntp_fixed"，不能作为 G1（非永久性）对照。
修复：前缀取 groups_ex[g][i][0]、答案取 groups_ex[g][j][1]。重跑见 ntp_randperm_v2。
