# 准备包验证记录（development）

日期：2026-10-16（Asia/Shanghai）  
范围：`outputs/sibling-confirmation-20261016/` 的结构门禁和现有 P6 工程回归。  
边界：没有生成 Sibling 数据、没有训练正式模型、没有读取服务器权重、没有运行留出评价。

## 通过

| 命令 | 结果 | 说明 |
|---|---|---|
| `env312/Scripts/python.exe code/validate_sibling_manifest.py freeze_manifest.json run_manifest.tsv` | PASS | 九个条目、三臂、三种子、批量身份和阻断状态一致 |
| `env312/Scripts/python.exe code/evaluator.py --selftest` | PASS | 评价器工程自检；不是 Sibling 结果 |
| `env312/Scripts/python.exe code/test_pipeline.py` | PASS (`ALL OK`) | 现有微型流水线语义回归；不是前瞻有效性证据 |
| `env312/Scripts/python.exe -m py_compile code/validate_sibling_manifest.py` | PASS | 新增结构检查脚本语法检查 |
| `bash -n code/run_stage0b.sh` 与 `py_compile`（evaluator/train/rescore/test） | PASS | launcher、显式图加载和身份字段语法检查 |
| 临时微型 GPT-2 + 临时数据 + 两个临时图 JSON，`train.py --task sibling --sibling_graph_dir ... --steps 2` | PASS | 确认显式图目录能进入训练评价循环并写入图文件 SHA；fixture 不是 Sibling 数据或结果 |

## 根仓库约定检查

按 `AGENTS.md` 运行了 `python tools/validate.py` 和
`python -m unittest discover -s tests -v`。两项仍被仓库既有完整性基线阻断，未修改历史材料来消除它们：

* `tools/validate.py`：三个既有 `runs/smoke/churn_analysis.json` 的 `NaN`、`archive/opd-v0.2.md` 的 archived bytes 变化、External review source bytes 变化。
* unittest：153 项中 4 failures、1 error；包括相同的 External review source/完整性问题、两个既有 patch registry 检查失败，以及 Windows 默认 GBK 读取生成材料的既有 `UnicodeDecodeError`。

这些失败不涉及本准备包的 JSON、TSV、验证脚本或 P6 训练/评价代码；提交前以完整日志和当前仓库状态复核。

上述 Sibling 接口 smoke 没有清除完整知识图、官方数据快照或正式步数门禁；它只证明当前 trainer 能接收显式 `task=sibling` 加完整图目录并完成临时工程 fixture 的最小循环。
