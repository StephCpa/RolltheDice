"""P6 WP0c: 数据生成——忠实移植自 chenwu98/algorithmic-creativity @7176dc7 的
circle-construction/ntp/circle.ipynb 与 sibling-discovery/ntp/sibling.ipynb /
sibling_no_hash.ipynb / circle_hybrid.ipynb（来源单元格逻辑逐行对照）。

偏差登记：
- DATA_ROOT 改为本地路径；
- 固定 numpy/python random 种子（原 notebook 未设种子），保证可复现；
- circle 测试探针数量、训练规模等参数保持原值（10000 训练、1024 探针；
  sibling 50000 训练、3000 探针）。
用法：python datagen.py --task circle|sibling --out ../../data
"""
import argparse
import json
import os
import random
import string
from copy import deepcopy

import numpy as np

SPECIAL_TOKENS = ["<mask>", "<sep>", "<a>", "</a>", "<q>", "</q>"]


def choose(arr, ratio_or_count, rng):
    if isinstance(ratio_or_count, float):
        num = round(ratio_or_count * len(arr))
    else:
        num = ratio_or_count
    if num >= len(arr):
        return arr
    inds = rng.choice(len(arr), num, replace=False).tolist()
    return [arr[i] for i in inds]


# ---------------- circle ----------------

def circle_form(hash_str, nodes):
    input_text = "".join([hash_str, "<q>"])
    edges = [nodes[i] + nodes[i + 1] for i in range(len(nodes) - 1)]
    edges = edges + [nodes[-1] + nodes[0]]
    random.shuffle(edges)
    target_text = input_text + "<sep>".join(edges) + "</a>"
    return {"input_text": input_text, "target_text": target_text}


def circle_form_test(hash_str):
    input_text = "".join([hash_str, "<q>"])
    return {"input_text": input_text, "target_text": input_text + "</a>"}


def circle_build(num_entities, num_nodes, hash_str_len, max_train_num=10000, n_test=1024):
    entities_vocab = ["<a_{}>".format(i) for i in range(num_entities)]
    chars = string.ascii_lowercase + string.digits
    base = len(chars)
    used_hashes = set()
    train_sequences, train_no_hash = [], []
    for _ in range(max_train_num):
        nodes = random.sample(entities_vocab, num_nodes)
        if hash_str_len == 0:
            hash_str = ""
        else:
            while True:
                hash_digits = [random.randint(0, base - 1) for _ in range(hash_str_len)]
                hash_str = "".join(chars[d] for d in hash_digits)
                if hash_str not in used_hashes:
                    used_hashes.add(hash_str)
                    break
        train_sequences.append(circle_form(hash_str, nodes))
        train_no_hash.append(circle_form("", nodes))
    test_sequences, test_no_hash = [], []
    for _ in range(n_test):
        if hash_str_len == 0:
            hash_str = ""
        else:
            while True:
                hash_digits = [random.randint(0, base - 1) for _ in range(hash_str_len)]
                hash_str = "".join(chars[d] for d in hash_digits)
                if hash_str not in used_hashes:
                    used_hashes.add(hash_str)
                    break
        test_sequences.append(circle_form_test(hash_str))
        test_no_hash.append(circle_form_test(""))
    return entities_vocab, train_sequences, test_sequences, train_no_hash, test_no_hash


# ---------------- sibling ----------------

def sib_form(hash_str, a, b1, b2):
    input_text = "".join([hash_str, "<q>"])
    target_text = input_text + "".join([b1, b2, a, "</a>"])
    return {"input_text": input_text, "target_text": target_text}


def sib_form_test(hash_str):
    input_text = "".join([hash_str, "<q>"])
    return {"input_text": input_text, "target_text": input_text + "</a>"}


def sibling_build(num_a, num_b_per_a, hash_str_len, keep_prob=0.995):
    entities_a = ["<a_{}>".format(i) for i in range(num_a)]
    entities_b1 = ["<b1_{}>".format(i) for i in range(num_b_per_a * num_a)]
    entities_b2 = ["<b2_{}>".format(i) for i in range(num_b_per_a * num_a)]
    entity_vocab = entities_a + entities_b1 + entities_b2
    b1_dict = {a: [entities_b1[i * num_b_per_a + j] for j in range(num_b_per_a)]
               for i, a in enumerate(entities_a)}
    b2_dict = {a: [entities_b2[i * num_b_per_a + j] for j in range(num_b_per_a)]
               for i, a in enumerate(entities_a)}
    chars = string.ascii_lowercase + string.digits
    base = len(chars)
    used_hashes = set()
    train_sequences, test_sequences = [], []
    for entity_a in entities_a:
        for b1 in b1_dict[entity_a]:
            for b2 in b2_dict[entity_a]:
                if hash_str_len == 0:
                    hash_str = ""
                else:
                    while True:
                        hash_digits = [random.randint(0, base - 1) for _ in range(hash_str_len)]
                        hash_str = "".join(chars[d] for d in hash_digits)
                        if hash_str not in used_hashes:
                            used_hashes.add(hash_str)
                            break
                if np.random.uniform() > (1.0 - keep_prob):
                    test_sequences.append(sib_form_test(hash_str))
                else:
                    train_sequences.append(sib_form(hash_str, entity_a, b1, b2))
    return entity_vocab, train_sequences, test_sequences


def remove_hash(s):
    q_idx = s.find("<q>")
    if q_idx == -1:
        return s
    return s[q_idx:]


# ---------------- 写盘 ----------------

def write_split(root, name, train_sequences, test_sequences, vocab, test_size, rng):
    out = os.path.join(root, name)
    os.makedirs(out, exist_ok=True)
    train_ds = choose(train_sequences, len(train_sequences), rng)  # 保持顺序（规模即全部）
    test_sequences = choose(test_sequences, test_size, rng)
    probes = []
    for item in choose(train_ds, test_size, rng):
        p = deepcopy(item)
        p["type"] = "train"
        probes.append(p)
    for item in test_sequences:
        p = deepcopy(item)
        p["type"] = "test"
        probes.append(p)
    with open(os.path.join(out, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_ds, f)
    with open(os.path.join(out, "valid.json"), "w", encoding="utf-8") as f:
        json.dump(test_sequences, f)
    with open(os.path.join(out, "test.json"), "w", encoding="utf-8") as f:
        json.dump(probes, f)
    with open(os.path.join(out, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    print(name, "train:", len(train_ds), "probes:", len(probes))


def make_hybrid(train_sequences, n_dummy):
    """移植 circle_hybrid.ipynb：每例 1×NTP + 3×MTP（dummy=eos pad）。
    n_dummy: circle 为 NUM_NODES*3=27；sibling 为 3（b1,b2,a）。"""
    pad_token = "<|endoftext|>"
    hybrid = []
    for seq in train_sequences:
        input_text, target_text = seq["input_text"], seq["target_text"]
        hybrid.append({"input_text": input_text, "target_text": target_text,
                       "target_labels": target_text})
        for _ in range(3):
            hybrid.append({"input_text": input_text,
                           "target_text": input_text + pad_token * n_dummy + "</a>",
                           "target_labels": target_text})
    return hybrid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["circle", "sibling"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261009)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.task == "circle":
        NUM_ENTITIES, NUM_NODES, HASH_LEN, TRAIN_SIZE = 10, 9, 10, 10000
        rng = np.random.default_rng(args.seed)
        ev, tr, te, tr_nh, te_nh = circle_build(NUM_ENTITIES, NUM_NODES, HASH_LEN)
        vocab = ev + SPECIAL_TOKENS
        write_split(args.out, f"circle.{NUM_ENTITIES}.{NUM_NODES}.{HASH_LEN}.{TRAIN_SIZE}",
                    tr, te, vocab, 1024, rng)
        write_split(args.out, f"circle.{NUM_ENTITIES}.{NUM_NODES}.0.{TRAIN_SIZE}",
                    tr_nh, te_nh, vocab, 1024, rng)
        # teacherless hybrid（基于 hash 版）
        hy = make_hybrid(tr, NUM_NODES * 3)
        out = os.path.join(args.out, f"circle_hybrid.{NUM_ENTITIES}.{NUM_NODES}.{HASH_LEN}.{TRAIN_SIZE}")
        os.makedirs(out, exist_ok=True)
        for fn in ["valid.json", "test.json", "vocab.json"]:
            src = os.path.join(args.out, f"circle.{NUM_ENTITIES}.{NUM_NODES}.{HASH_LEN}.{TRAIN_SIZE}", fn)
            with open(src, encoding="utf-8") as f:
                data = json.load(f)
            with open(os.path.join(out, fn), "w", encoding="utf-8") as f:
                json.dump(data, f)
        with open(os.path.join(out, "train.json"), "w", encoding="utf-8") as f:
            json.dump(hy, f)
        print("circle_hybrid train:", len(hy))
    else:
        NUM_A, NUM_B, HASH_LEN, TRAIN_SIZE = 10, 1000, 10, 50000
        rng = np.random.default_rng(args.seed)
        ev, tr, te = sibling_build(NUM_A, NUM_B, HASH_LEN)
        vocab = ev + SPECIAL_TOKENS
        write_split(args.out, f"sibling.{NUM_A}.{NUM_B}.{HASH_LEN}.{TRAIN_SIZE}",
                    tr, te, vocab, 3000, rng)
        no_hash_tr = [{"input_text": remove_hash(s["input_text"]),
                       "target_text": remove_hash(s["target_text"])} for s in tr]
        no_hash_te = [{"input_text": remove_hash(s["input_text"]),
                       "target_text": remove_hash(s["target_text"])} for s in te]
        write_split(args.out, f"sibling.{NUM_A}.{NUM_B}.0.{TRAIN_SIZE}",
                    no_hash_tr, no_hash_te, vocab, 3000, rng)
        hy = make_hybrid(tr, 3)
        out = os.path.join(args.out, f"sibling_hybrid.{NUM_A}.{NUM_B}.{HASH_LEN}.{TRAIN_SIZE}")
        os.makedirs(out, exist_ok=True)
        for fn in ["valid.json", "test.json", "vocab.json"]:
            src = os.path.join(args.out, f"sibling.{NUM_A}.{NUM_B}.{HASH_LEN}.{TRAIN_SIZE}", fn)
            with open(src, encoding="utf-8") as f:
                data = json.load(f)
            with open(os.path.join(out, fn), "w", encoding="utf-8") as f:
                json.dump(data, f)
        with open(os.path.join(out, "train.json"), "w", encoding="utf-8") as f:
            json.dump(hy, f)
        print("sibling_hybrid train:", len(hy))


if __name__ == "__main__":
    main()
