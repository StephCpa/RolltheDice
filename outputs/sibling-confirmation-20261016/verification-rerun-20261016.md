# 准备包复核记录（development）

日期：2026-10-16（Asia/Shanghai）  
代码版本：`b2c995f`（文档改动提交前的代码版本）  
范围：显式 Sibling 知识图门禁、冻结 manifest 和现有 P6 工程回归；没有启动正式训练、没有读取服务器权重、没有运行留出评价。

## 本次复核

| 命令 | 结果 |
|---|---|
| `P6-Roll_the_dice/env312/Scripts/python.exe code/test_pipeline.py` | PASS (`ALL OK`) |
| `P6-Roll_the_dice/env312/Scripts/python.exe code/evaluator.py --selftest` | PASS (`SELFTEST OK`) |
| `P6-Roll_the_dice/env312/Scripts/python.exe code/validate_sibling_manifest.py outputs/sibling-confirmation-20261016/freeze_manifest.json outputs/sibling-confirmation-20261016/run_manifest.tsv` | PASS（9 个 blocked runs） |
| `P6-Roll_the_dice/env312/Scripts/python.exe -m py_compile code/evaluator.py code/train.py code/rescore.py code/test_pipeline.py code/validate_sibling_manifest.py` | PASS |
| `bash -n code/run_stage0b.sh` | PASS |

这组检查只确认代码接口和准备包结构，没有把临时 fixture 的输出升级为 Sibling 结果。数据快照、完整知识图、变体 SHA/不相交性和正式步数核价仍未完成，因此 manifest 继续保持 `blocked_pending_artifacts`。

根仓库的完整性基线失败及其未修改的历史来源，沿用 [`verification-20261016.md`](verification-20261016.md) 的记录；本次没有重写历史材料来消除这些失败。
