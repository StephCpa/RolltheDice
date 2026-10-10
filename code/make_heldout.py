"""生成留出探针集：与训练集、valid.json、test.json 中所有 hash 均不相交的新 hash。

评审第 4 轮协议：test.json 前 128 个测试探针（训练中每 5000 步评价所用）是开发集，用于选检查点与
相位分析；最终报告在本脚本生成的留出集上一次性评测（rescore.py --probe_file）。
随机流独立于 datagen（不改变任何已有数据）。

用法：python code/make_heldout.py --data_dir data/circle.10.9.10.10000 --n 2048
输出：<data_dir>/heldout.json（与 test.json 中 type=test 条目同格式）并打印 sha256。
"""
import argparse
import hashlib
import json
import os
import random
import string


def hash_of(s):
    return s.split("<q>")[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    used = set()
    lens = set()
    for fn in ("train.json", "valid.json", "test.json"):
        p = os.path.join(args.data_dir, fn)
        if os.path.exists(p):
            for s in json.load(open(p, encoding="utf-8")):
                h = hash_of(s["input_text"])
                used.add(h)
                lens.add(len(h))
    hash_len = max(lens) if lens else 0
    chars = string.ascii_lowercase + string.digits
    rnd = random.Random(args.seed)
    out = []
    if hash_len == 0:   # 无 hash 数据：所有探针都是同一个 prompt
        out = [{"input_text": "<q>", "target_text": "<q></a>", "type": "test"}
               for _ in range(args.n)]
    else:
        new = set()
        while len(new) < args.n:
            h = "".join(rnd.choice(chars) for _ in range(hash_len))
            if h not in used and h not in new:
                new.add(h)
                out.append({"input_text": h + "<q>", "target_text": h + "<q></a>", "type": "test"})
    path = args.out or os.path.join(args.data_dir, "heldout.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(json.dumps({"out": path, "n": len(out), "hash_len": hash_len,
                      "n_excluded_hashes": len(used), "seed": args.seed, "sha256": sha}))


if __name__ == "__main__":
    main()
