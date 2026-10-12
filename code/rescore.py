"""对已保存的模型（final_model 或 ckpts/stepN）重新生成并评分（不训练）。

用途：
- 评价器修订后重评已有模型（事件 #4/#5）；
- 检查点选择协议：开发集（训练中 test.json 前 128 个测试探针）选出的检查点，在留出集
  （make_heldout.py 生成的 heldout.json）上一次性评测；
- memo_own：按本臂实际训练配对评训练对复现（--pairing_file；缺省读 model_dir/pairing.json，
  均无则视为名义配对——对 setc 旧模型请传 runs/.../pistar_final.json）；
- 无 seed 臂用温度采样作公平对照（greedy + 恒定 prompt 必然只有 1 个输出）。
输出：<out>/scores.jsonl 追加一行（step 字段=--step 标签）、preds_step<step>.jsonl、rescore_meta.json。

例：
  python code/rescore.py --model_dir runs/x/ckpts/step20000 --data_dir data/circle.10.9.10.10000 \
      --probe_file data/circle.10.9.10.10000/heldout.json --out runs/x/heldout --step 20000
  python code/rescore.py --model_dir runs/server-circle-40k/setc_m4_v2/final_model \
      --data_dir data/circle.10.9.10.10000 --pairing_file runs/server-circle-40k/setc_m4_v2/pistar_final.json \
      --out runs/server-circle-40k/setc_m4_v2/rescore_memo --n_test 128 --step 40000
  python code/rescore.py --model_dir runs/noseed/final_model --data_dir data/circle.10.9.0.10000 \
      --out runs/noseed/rescore_T1 --sample --temp 1.0
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_pairing, run_eval  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--task", choices=["circle", "sibling"], default="circle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe_file", default=None,
                    help="测试探针来源（缺省 data_dir/test.json 的 type=test 条目）")
    ap.add_argument("--n_test", type=int, default=None, help="测试探针数（缺省=来源中全部）")
    ap.add_argument("--n_train", type=int, default=128, help="训练探针数（取自 test.json 的 type=train）")
    ap.add_argument("--train_offset", type=int, default=0, help="训练探针起始偏移（避开开发集用 128）")
    ap.add_argument("--n_probes", type=int, default=None,
                    help="旧接口：train/test 各取一半（给定时覆盖 --n_test/--n_train）")
    ap.add_argument("--pairing_file", default=None,
                    help="本臂实际训练配对（memo_own）；缺省 model_dir/pairing.json，再缺省名义")
    ap.add_argument("--n_samples", type=int, default=1, help="每探针生成数")
    ap.add_argument("--sample", action="store_true", help="温度采样（默认 greedy）")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--step", type=int, default=0, help="写入 scores/preds 的步数标签")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained(args.model_dir)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model_dir).to(device)
    with open(os.path.join(args.data_dir, "train.json"), encoding="utf-8") as f:
        raw = json.load(f)
    with open(os.path.join(args.data_dir, "test.json"), encoding="utf-8") as f:
        probes_all = json.load(f)
    src = probes_all
    if args.probe_file:
        with open(args.probe_file, encoding="utf-8") as f:
            src = json.load(f)
    n_test, n_train = args.n_test, args.n_train
    if args.n_probes is not None:
        n_test = n_train = args.n_probes // 2
    te = [p for p in src if p.get("type", "test") == "test"]
    te = te[:n_test] if n_test is not None else te
    tr = [p for p in probes_all if p.get("type") == "train"][args.train_offset:args.train_offset + n_train]

    pf = args.pairing_file
    if pf is None and os.path.exists(os.path.join(args.model_dir, "pairing.json")):
        pf = os.path.join(args.model_dir, "pairing.json")
    text2idx = {s["input_text"]: k for k, s in enumerate(raw)}
    own_targets = None
    if len(text2idx) == len(raw):   # 无 hash 数据没有逐 seed 配对
        try:
            R = load_pairing(pf, len(raw)) if pf else list(range(len(raw)))
        except AssertionError as e:   # 配对来自另一份训练数据（如 dup_hash 变体）：不评 memo_own
            print(f"[warn] 配对与 --data_dir 不匹配，跳过 memo_own：{e}")
            R = None
        if R is not None:
            own_targets = {p["input_text"]: raw[R[text2idx[p["input_text"]]]]["target_text"]
                           for p in tr if p["input_text"] in text2idx}

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "rescore_meta.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "pairing_used": pf or "nominal",
                   "n_train_probes": len(tr), "n_test_probes": len(te)}, f, indent=2)
    run_eval(model, tok, tr + te, raw, args.task, args.out, args.step, device,
             args.n_samples, args.temp, tok.eos_token_id, save_preds=True,
             do_sample=args.sample, own_targets=own_targets,
             extra={"model_dir": args.model_dir, "probe_file": args.probe_file or "test.json"})


if __name__ == "__main__":
    main()
