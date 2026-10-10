"""对已保存的 final_model 重新生成并评分（不训练）。

用途：评价器修订后重评已有模型；扩大探针数（test.json 中 train/test 各 1024 条，
训练时默认只用前 128 条）；无 seed 臂用温度采样作公平对照（greedy + 恒定 prompt
必然只有 1 个输出）。输出与训练时相同：<out>/scores.jsonl 追加一行、preds_step<step>.jsonl。

例：
  python code/rescore.py --model_dir runs/x/final_model --data_dir data/circle.10.9.10.10000 \
      --out runs/x/rescore --n_probes 2048
  python code/rescore.py --model_dir runs/noseed/final_model --data_dir data/circle.10.9.0.10000 \
      --out runs/noseed/rescore_T1 --n_probes 2048 --sample --temp 1.0
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import run_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--task", choices=["circle", "sibling"], default="circle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_probes", type=int, default=2048, help="train/test 各取一半")
    ap.add_argument("--n_samples", type=int, default=1, help="每探针生成数")
    ap.add_argument("--sample", action="store_true", help="温度采样（默认 greedy）")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--step", type=int, default=0, help="写入 scores/preds 的步数标签")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

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
    tr = [p for p in probes_all if p.get("type") == "train"][: args.n_probes // 2]
    te = [p for p in probes_all if p.get("type") == "test"][: args.n_probes // 2]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "rescore_meta.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "n_train_probes": len(tr), "n_test_probes": len(te)}, f, indent=2)
    run_eval(model, tok, tr + te, raw, args.task, args.out, args.step, device,
             args.n_samples, args.temp, tok.eos_token_id, save_preds=True,
             do_sample=args.sample)


if __name__ == "__main__":
    main()
