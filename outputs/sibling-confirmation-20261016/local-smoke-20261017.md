# Local Sibling engineering smoke — 2026-10-17

状态：development engineering check；不属于 Sibling 科学结果。

## 命令

使用 P6 `env312`、本地 RTX 2080 Ti、候选 public 5×500 base 数据、
`task=sibling`、`ntp_fixed`、2 steps、batch 4、4 probes、`--no_save_preds`。
模型参数为 `--model gpt2 --model_revision local-gpt2-smoke`。

## 结果

训练进程在加载 GPT-2 tokenizer 时终止。Hugging Face Hub 请求返回
`httpx.ConnectTimeout: [WinError 10060]`；本地缓存不足以完成 `GPT2TokenizerFast.from_pretrained("gpt2")`。
没有生成 run directory、checkpoint、预测或 heldout 输出。

## 边界

该失败只说明当前本地环境无法完成依赖模型加载，不说明 Sibling 方法、候选数据或评价器失败。
没有因为网络超时重跑，也没有将本次检查计入训练结果或资源预算。
