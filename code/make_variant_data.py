"""数据变体：目标多重性实验的混杂对照（plan/icml2027-plan.md §4.1b）。

dup_hash（"每答案 k 个 hash、每 hash 单目标"）：为每个训练目标再配 k-1 个新 hash。
与 ntp_2tgt（每 hash 两目标、每答案两 hash）在 (hash, 答案) 对总数上匹配，但每个 hash 仍只决定一个答案——
用来区分"每个 seed 有两个目标"与"每个答案有两个 seed / 训练对数翻倍"。
新 hash 与原 train/valid/test/heldout 的全部 hash 不相交；原训练样本保持原顺序与编号（新条目追加在后），
test.json / valid.json / vocab.json / heldout.json 原样复制——开发集与留出集与基础数据完全相同。

用法：python code/make_variant_data.py --base_dir data/circle.10.9.10.10000 --mode dup_hash --k 2 \
        --out data/circle.10.9.10.10000.dup2
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import string


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["dup_hash"], default="dup_hash")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261012)
    args = ap.parse_args(argv)
    assert args.k >= 2

    train = json.load(open(os.path.join(args.base_dir, "train.json"), encoding="utf-8"))
    used = set()
    for fn in ("train.json", "valid.json", "test.json", "heldout.json"):
        p = os.path.join(args.base_dir, fn)
        if os.path.exists(p):
            used |= {s["input_text"].split("<q>")[0] for s in json.load(open(p, encoding="utf-8"))}
    hash_len = max(len(s["input_text"].split("<q>")[0]) for s in train)
    assert hash_len > 0, "dup_hash 需要 hash 条件数据"
    chars = string.ascii_lowercase + string.digits
    rnd = random.Random(args.seed)
    extra = []
    for s in train:
        answer = s["target_text"].split("<q>", 1)[1]
        for _ in range(args.k - 1):
            while True:
                h = "".join(rnd.choice(chars) for _ in range(hash_len))
                if h not in used:
                    used.add(h)
                    break
            extra.append({"input_text": h + "<q>", "target_text": h + "<q>" + answer})
    os.makedirs(args.out, exist_ok=True)
    out_train = train + extra
    with open(os.path.join(args.out, "train.json"), "w", encoding="utf-8") as f:
        json.dump(out_train, f)
    for fn in ("test.json", "valid.json", "vocab.json", "heldout.json"):
        p = os.path.join(args.base_dir, fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(args.out, fn))
    sha = hashlib.sha256(open(os.path.join(args.out, "train.json"), "rb").read()).hexdigest()
    meta = {"mode": args.mode, "k": args.k, "base_dir": args.base_dir, "seed": args.seed,
            "n_train_base": len(train), "n_train_out": len(out_train), "train_sha256": sha}
    with open(os.path.join(args.out, "variant.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
