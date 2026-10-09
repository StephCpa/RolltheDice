"""P6 WP0c: 统一评价器——移植 circle/sibling 的 eval_qa.py @7176dc7。

指标定义（与原文一致）：
- validity: circle=每节点度恰为2的单一连通环；sibling=(b1,a)与(b2,a)均为知识图真边；
- seen: circle=按首次出现重标定后的结构级匹配；sibling=原始串精确匹配；
- unseen = valid 且非 seen；creativity=有效且未见的去重比例；diversity=有效的去重比例；
- train_memorization = 与 gold 逐串一致。
自检（--selftest）对构造样例独立复算，属 WP0c 评价核验。
"""
import argparse
import json
import os
import sys


def complete(s):
    if not s.endswith(">"):
        s += ">"
    if not s.startswith("<"):
        s = "<" + s
    return s


# ---------- circle ----------

def circle_canonicalize(seq):
    if seq.endswith("</a>"):
        seq = seq[:-4]
    edges = seq.split("<sep>")
    edges = [e.split("><") for e in edges]
    if not all(len(e) == 2 for e in edges):
        return None
    edges = [(complete(e[0]), complete(e[1])) for e in edges]
    canon, n2c = [], {}
    for n1, n2 in edges:
        n2c.setdefault(n1, len(n2c))
        n2c.setdefault(n2, len(n2c))
        canon.append((n2c[n1], n2c[n2]))
    return "<sep>".join("<{}><{}>".format(a, b) for a, b in canon)


def circle_validity(pred):
    assert pred.count("</a>") == 1, "预测必须恰含一个</a>"
    pred = pred.split("<q>")[1].split("</a>")[0]
    try:
        edges = []
        for e in pred.split("<sep>"):
            n1, n2 = e.split("><")
            edges.append((complete(n1), complete(n2)))
    except Exception:
        return 0
    if not edges:
        return 0
    node_counts = {}
    for n1, n2 in edges:
        node_counts[n1] = node_counts.get(n1, 0) + 1
        node_counts[n2] = node_counts.get(n2, 0) + 1
    if not all(c == 2 for c in node_counts.values()):
        return 0
    adj = {}
    for n1, n2 in edges:
        adj.setdefault(n1, []).append(n2)
        adj.setdefault(n2, []).append(n1)
    visited = set()
    start = list(adj.keys())[0]
    curr, prev = start, None
    while curr not in visited:
        visited.add(curr)
        nxt = None
        for nb in adj[curr]:
            if nb != prev and nb not in visited:
                nxt = nb
                break
        if nxt is None:
            if start in adj[curr] and len(visited) == len(node_counts):
                return 1
            return 0
        prev, curr = curr, nxt
    return 0


# ---------- sibling ----------

def build_sibling_graph(train_sequences):
    """从训练数据重建知识图：b1_dict[a] / b2_dict[a]。"""
    b1_dict, b2_dict = {}, {}
    for seq in train_sequences:
        ans = seq["target_text"].split("<q>")[1]
        if ans.endswith("</a>"):
            ans = ans[:-4]
        parts = ans.split("><")
        if len(parts) != 3:
            continue
        b1, b2, a = complete(parts[0]), complete(parts[1]), complete(parts[2])
        b1_dict.setdefault(a, set()).add(b1)
        b2_dict.setdefault(a, set()).add(b2)
    return b1_dict, b2_dict


class Evaluator:
    def __init__(self, task, train_sequences):
        self.task = task
        if task == "circle":
            self.train_answers = set()
            for s in train_sequences:
                c = circle_canonicalize(s["target_text"].split("<q>")[1])
                if c is not None:
                    self.train_answers.add(c)
        else:
            self.b1_dict, self.b2_dict = build_sibling_graph(train_sequences)
            self.train_answers = set(
                s["target_text"].split("<q>")[1].split("</a>")[0] for s in train_sequences)

    def seen(self, pred):
        if "<q>" not in pred:
            return 0
        ans = pred.split("<q>")[1]
        if self.task == "circle":
            return int(circle_canonicalize(ans) in self.train_answers)
        return int(ans.split("</a>")[0] in self.train_answers)

    def validity(self, pred):
        if pred.count("</a>") != 1 or "<q>" not in pred:
            return 0  # 畸形输出（未停于</a>等）计为无效，与原 assert 的差别已登记
        if self.task == "circle":
            return circle_validity(pred)
        ans = pred.split("<q>")[1].split("</a>")[0]
        try:
            b1, b2, a = ans.split("><")
        except Exception:
            return 0
        b1, b2, a = complete(b1), complete(b2), complete(a)
        try:
            return int((b1 in self.b1_dict.get(a, ())) and (b2 in self.b2_dict.get(a, ())))
        except Exception:
            return 0

    def eval_items(self, all_items, n_gen=1):
        """all_items: [{type, model_output(可为多采样列表), target_text}]。
        返回汇总分数。多样本时对每样本计分后取均值（creativity/diversity 用并集）。"""
        acc = {"train_memorization": [], "test_seen": [], "test_unseen": [],
               "test_validity": []}
        valid_answers, valid_novel_answers = set(), set()
        for item in all_items:
            t = item.get("type", "test")
            outs = item["model_output"] if isinstance(item["model_output"], list) \
                else [item["model_output"]] * n_gen
            for pred in outs:
                if t == "train":
                    gold = item["target_text"].split("</a>")[0]
                    acc["train_memorization"].append(int(pred.split("</a>")[0] == gold))
                else:
                    sv = self.seen(pred)
                    vd = self.validity(pred)
                    acc["test_seen"].append(sv)
                    acc["test_validity"].append(vd)
                    acc["test_unseen"].append(0 if sv == 1 else (1 if vd == 1 else 0))
                    if vd == 1:
                        ans = pred.split("<q>")[1].split("</a>")[0]
                        key = circle_canonicalize(ans) if self.task == "circle" else ans
                        valid_answers.add(key)
                        if sv == 0:
                            valid_novel_answers.add(key)
        n_test = max(len(acc["test_seen"]), 1)
        return {
            "train_memorization": sum(acc["train_memorization"]) / max(len(acc["train_memorization"]), 1),
            "test_seen": sum(acc["test_seen"]) / n_test,
            "test_validity": sum(acc["test_validity"]) / n_test,
            "test_unseen": sum(acc["test_unseen"]) / n_test,
            "n_unique_valid": len(valid_answers),
            "n_unique_valid_novel": len(valid_novel_answers),
        }


def selftest():
    """WP0c 评价核验：构造样例独立复算。"""
    ok = True

    def chk(name, got, want):
        nonlocal ok
        flag = got == want
        ok = ok and flag
        print(("PASS" if flag else "FAIL"), name, "got", got, "want", want)

    # circle 合法环：9 节点
    nodes = [f"<a_{i}>" for i in range(9)]
    edges = [nodes[i] + nodes[i + 1] for i in range(8)] + [nodes[8] + nodes[0]]
    pred = "abcdefghij<q>" + "<sep>".join(edges) + "</a>"
    chk("circle valid 9-cycle", circle_validity(pred), 1)
    # 度数违规：重复一条边、去掉一条
    bad = "abcdefghij<q>" + "<sep>".join(edges[:8]) + "</a>"
    chk("circle invalid path", circle_validity(bad), 0)
    dup = "abcdefghij<q>" + "<sep>".join(edges + [edges[0]]) + "</a>"
    chk("circle invalid dup edge", circle_validity(dup), 0)
    # 两个分离环（度都对但不连通）
    n4 = [f"<a_{i}>" for i in range(4)]
    c1 = [n4[i] + n4[(i + 1) % 4] for i in range(4)]
    n5 = [f"<a_{i}>" for i in range(4, 9)]
    c2 = [n5[i] + n5[(i + 1) % 5] for i in range(5)]
    two = "hhhh<q>" + "<sep>".join(c1 + c2) + "</a>"
    chk("circle invalid disconnected", circle_validity(two), 0)
    # 边序打乱仍有效
    import random
    sh = edges[:]
    random.shuffle(sh)
    chk("circle valid shuffled", circle_validity("x<q>" + "<sep>".join(sh) + "</a>"), 1)

    # canonicalize：重标定不变性（同一结构、不同实体名与边序 → 同 canonical?）
    c1 = circle_canonicalize("<a_1><a_2><sep><a_2><a_3></a>")
    c2 = circle_canonicalize("<a_7><a_4><sep><a_4><a_9></a>")
    chk("canonicalize relabel-invariant", c1 == c2, True)
    # seen 判定：换名同结构 → seen
    tr = [{"target_text": "h<q><a_1><a_2><sep><a_2><a_3></a>"}]
    ev = Evaluator("circle", tr)
    chk("circle seen relabeled", ev.seen("zz<q><a_5><a_6><sep><a_6><a_9></a>"), 1)
    chk("circle unseen structure", ev.seen("zz<q><a_1><a_2><sep><a_2><a_3><sep><a_3><a_1></a>"), 0)

    # sibling
    tr_s = [{"target_text": "h<q><b1_1><b2_2><a_0></a>"},
            {"target_text": "h<q><b1_5><b2_9><a_3></a>"}]
    evs = Evaluator("sibling", tr_s)
    chk("sibling valid known edges", evs.validity("q<q><b1_1><b2_2><a_0></a>"), 1)
    chk("sibling invalid wrong parent", evs.validity("q<q><b1_1><b2_2><a_3></a>"), 0)
    chk("sibling unknown node", evs.validity("q<q><b1_99><b2_2><a_0></a>"), 0)
    chk("sibling seen exact", evs.seen("q<q><b1_5><b2_9><a_3></a>"), 1)
    chk("sibling unseen valid", evs.seen("q<q><b1_5><b2_2><a_3></a>"), 0)
    print("SELFTEST", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["circle", "sibling"], required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--pred", required=True, help="预测 json（[{type,model_output,target_text}]）")
    args = ap.parse_args()
    with open(os.path.join(args.data_dir, "train.json"), encoding="utf-8") as f:
        tr = json.load(f)
    with open(args.pred, encoding="utf-8") as f:
        items = json.load(f)
    ev = Evaluator(args.task, tr)
    print(json.dumps(ev.eval_items(items), indent=2))
