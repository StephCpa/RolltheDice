# Sibling 远端 artifact 只读检索记录（development）

日期：2026-10-17  
来源版本：远端 `gpu-gfkd:/home/wlwuser/P6-Roll_the_dice`；通过只读 SSH 检索，最近一次状态轮询时间为 2026-10-17 06:06:55（Asia/Shanghai）。  
范围：检查是否存在同时包含 `train.json`、`valid.json`、`test.json`、`heldout.json` 的 Sibling 数据目录，以及完整实体图文件。未复制、修改或启动训练。

## 观察

远端未发现同时包含四个分割的单一 `sibling*` 数据目录。可见的派生候选为：

| 路径 | 文件 | 大小（字节） | SHA256 |
|---|---|---:|---|
| `data/sibling.derived-5x500.20261017/base/` | `train.json` | 4,355,872 | `8fe58f245afefea6c996fca09ef3151017b10e92b8d8f3568f4e10fba0c8b04` |
| 同上 | `valid.json` | 195,001 | `6ce723f717f51d8f31250bd120362a23f31ae4985ee967ec9f47c89a46f8296a` |
| 同上 | `test.json` | 543,325 | `34700619ddbe06594d8c52ce9c53d37307a335204bd3339cd00621b7ee4baa47` |
| `data/sibling.derived-5x500.20261017/dup_hash_k2/` | `train.json` | 9,111,742 | `fee377335f284b829987ba9ee9ca61590a560d288b7fc77ce176b0373e7c77a4` |
| `data/sibling.derived-5x500.20261017/graph/` | `entities_b1_dict.json` | 28,941 | `9977d5b45a5692d8af63e1c7b33745e7b0ec498ce287748f4ea00650399a346d` |
| 同上 | `entities_b2_dict.json` | 28,941 | `7c66efda12ea3adee6a0d83788f06ae096ca56df3582bd3a4b7d093deb089f41` |

`dup_hash_k2` 的 valid/test 与 base 相同；该目录同样没有 `heldout.json`。以上路径属于公开派生候选，仍受 `freeze_manifest.json` 的 `blocked_pending_artifacts` 约束。

## 判定

本次检索不能解除 Sibling 正式运行门禁。仍缺少官方数据快照身份和独立 heldout；实体图虽存在于派生候选目录，尚不足以替代完整、已冻结的 Sibling artifact。后续若采用该候选，需先由项目负责人接受其数据契约，重新登记来源、排除集合、SHA 和 heldout 生成规则。
