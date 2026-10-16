"""P6 WP0c: 统一评价器——移植 circle/sibling 的 eval_qa.py @7176dc7。

指标定义（与原文一致）：
- validity: circle=每节点度恰为2的单一连通环；sibling=(b1,a)与(b2,a)均为知识图真边；
- seen: circle=按首次出现重标定后的结构级匹配；sibling=原始串精确匹配；
- unseen = valid 且非 seen；creativity=有效且未见的去重比例；diversity=有效的去重比例；
- train_memorization = 与 gold 逐串一致（即复现数据集的 hash↔target 配对；
  对自选配对的 setc 臂，它不等于"是否记忆训练集"）。
与原文的差异：n_unique_valid / n_unique_valid_novel 按 canonical 结构去重；
原 eval_qa 的 creativity/diversity 按原始答案串去重并除以生成数。
自检（--selftest）对构造样例独立复算，属 WP0c 评价核验。
"""
import argparse
import json
import os
import sys
import re


# Stable identity recorded by rescore.py.  Keep this separate from the source
# hash: the hash detects an uncommitted edit, while this label identifies the
# evaluator contract used in a result package.
EVALUATOR_VERSION = "p6-evaluator-v2"
SIBLING_GRAPH_SCHEMA_VERSION = "p6-sibling-complete-v1"


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


def line_validity(pred):
    """Line Construction 有效性（移植自原仓库 line-construction/ntp/eval_qa.py @7176dc7）：
    恰 2 个节点度 1（端点）、其余度 2，且从端点遍历覆盖全部节点（单一连通路径）。"""
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
    n_once = sum(1 for c in node_counts.values() if c == 1)
    if n_once != 2 or any(c != 2 for c in node_counts.values() if c != 1):
        return 0
    adj = {}
    for n1, n2 in edges:
        adj.setdefault(n1, []).append(n2)
        adj.setdefault(n2, []).append(n1)
    start = next(n for n, c in node_counts.items() if c == 1)
    visited, prev, curr = set(), None, start
    while curr not in visited:
        visited.add(curr)
        nxt = next((nb for nb in adj[curr] if nb != prev and nb not in visited), None)
        if nxt is None:
            return int(len(visited) == len(node_counts))
        prev, curr = curr, nxt
    return 0


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


def load_sibling_graph(graph_dir):
    """Load the complete Sibling graph from explicit JSON artifacts.

    The train-derived graph remains available for legacy unit fixtures, but
    formal Sibling entry points must pass this explicit graph and record its
    identity.  Values are normalized to sets so evaluation is independent of
    JSON list order.
    """
    if not graph_dir:
        raise ValueError("Sibling evaluation requires an explicit graph directory")
    paths = {name: os.path.join(graph_dir, name) for name in
             ("entities_b1_dict.json", "entities_b2_dict.json")}
    data = {}
    parent_re = re.compile(r"^<a_[0-9]+>$")
    value_re = {
        "entities_b1_dict.json": re.compile(r"^<b1_[0-9]+>$"),
        "entities_b2_dict.json": re.compile(r"^<b2_[0-9]+>$"),
    }
    for kind, path in paths.items():
        try:
            with open(path, encoding="utf-8") as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load Sibling graph artifact {path}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"Sibling graph artifact {path} must contain an object")
        if not obj:
            raise ValueError(f"Sibling graph artifact {path} must not be empty")
        normalized = {}
        for a, values in obj.items():
            if not isinstance(a, str) or not parent_re.fullmatch(a):
                raise ValueError(f"Sibling graph artifact {path} has invalid parent key {a!r}")
            if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
                raise ValueError(f"Sibling graph artifact {path} has invalid values for {a!r}")
            if len(set(values)) != len(values):
                raise ValueError(f"Sibling graph artifact {path} has duplicate values for {a!r}")
            if not all(value_re[kind].fullmatch(v) for v in values):
                raise ValueError(f"Sibling graph artifact {path} has values outside its b-family for {a!r}")
            normalized[a] = set(values)
        data[kind] = normalized
    if set(data["entities_b1_dict.json"]) != set(data["entities_b2_dict.json"]):
        raise ValueError("Sibling b1/b2 graph artifacts have different parent keys")
    return data["entities_b1_dict.json"], data["entities_b2_dict.json"]


class Evaluator:
    def __init__(self, task, train_sequences, sibling_graph=None):
        self.task = task
        if task in ("circle", "line"):
            # line 与 circle 同用首次出现重标定规范化（函数任务无关，见 circle_canonicalize）
            self.train_answers = set()
            for s in train_sequences:
                c = circle_canonicalize(s["target_text"].split("<q>")[1])
                if c is not None:
                    self.train_answers.add(c)
        else:
            if sibling_graph is None:
                self.b1_dict, self.b2_dict = build_sibling_graph(train_sequences)
            else:
                self.b1_dict, self.b2_dict = sibling_graph
            self.train_answers = set(
                s["target_text"].split("<q>")[1].split("</a>")[0] for s in train_sequences)

    def seen(self, pred):
        if "<q>" not in pred:
            return 0
        # 先截断到首个 </a>：批量 generate 中提前停止的行会被补 pad（<|endoftext|>），
        # 不截断则 circle_canonicalize 返回 None → 已见解被误判为"未见/新解"（事件 #4）。
        # 原实现 predict() 已对输出做 split("</a>")[0]+"</a>"，故原 eval_qa 无此问题。
        ans = pred.split("<q>")[1].split("</a>")[0]
        if self.task in ("circle", "line"):
            return int(circle_canonicalize(ans) in self.train_answers)
        return int(ans in self.train_answers)

    def validity(self, pred):
        if pred.count("</a>") != 1 or "<q>" not in pred:
            return 0  # 畸形输出（未停于</a>等）计为无效，与原 assert 的差别已登记
        if self.task == "circle":
            return circle_validity(pred)
        if self.task == "line":
            return line_validity(pred)
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
        """all_items: [{type, model_output(可为多采样列表), target_text[, own_target_text]}]。
        返回汇总分数。多样本时对每样本计分后取均值（creativity/diversity 用并集）。

        训练探针（type=train）：
        - train_memorization：与 target_text（数据集名义配对）逐串一致（原指标，保留）；
        - memo_own：仅比较答案部分（<q> 之后、</a> 之前）与 own_target_text 一致——
          own_target_text 为该臂实际训练所用配对的目标（setc 为当前/冻结 π*，名义臂为名义，
          --pairing_file 臂为回放配对）。注意不能比较整串：整串含各自的 hash 前缀，
          只可能与本 seed 的名义目标相等（review_analyses 首版即因此失效，事件 #5）；
        - train_seen：输出结构属于训练集任一目标（结构级，等价于"复现某个训练目标"）。
        测试探针：
        - test_unseen（=test_novel_valid）：有效且结构未见；
        - creativity：去重（canonical）后的有效新结构数 / 测试生成数——论文主指标口径；
        - creativity_raw / diversity_raw：原 eval_qa 口径（按原始答案串去重 / 测试生成数）。"""
        acc = {"train_memorization": [], "test_seen": [], "test_unseen": [],
               "test_validity": [], "memo_own": [], "train_seen": [], "train_validity": []}
        valid_answers, valid_novel_answers = set(), set()
        raw_valid, raw_novel = set(), set()
        for item in all_items:
            t = item.get("type", "test")
            outs = item["model_output"] if isinstance(item["model_output"], list) \
                else [item["model_output"]] * n_gen
            for pred in outs:
                if t == "train":
                    gold = item["target_text"].split("</a>")[0]
                    acc["train_memorization"].append(int(pred.split("</a>")[0] == gold))
                    acc["train_seen"].append(self.seen(pred))
                    acc["train_validity"].append(self.validity(pred))
                    own = item.get("own_target_text")
                    if own is not None:
                        acc["memo_own"].append(int(answer_of(pred) == answer_of(own)))
                else:
                    sv = self.seen(pred)
                    vd = self.validity(pred)
                    acc["test_seen"].append(sv)
                    acc["test_validity"].append(vd)
                    acc["test_unseen"].append(0 if sv == 1 else (1 if vd == 1 else 0))
                    if vd == 1:
                        ans = pred.split("<q>")[1].split("</a>")[0]
                        key = circle_canonicalize(ans) if self.task in ("circle", "line") else ans
                        valid_answers.add(key)
                        raw_valid.add(ans)
                        if sv == 0:
                            valid_novel_answers.add(key)
                            raw_novel.add(ans)
        n_test = max(len(acc["test_seen"]), 1)

        def mean(k):
            return sum(acc[k]) / len(acc[k]) if acc[k] else None

        out = {
            "train_memorization": sum(acc["train_memorization"]) / max(len(acc["train_memorization"]), 1),
            "test_seen": sum(acc["test_seen"]) / n_test,
            "test_validity": sum(acc["test_validity"]) / n_test,
            "test_unseen": sum(acc["test_unseen"]) / n_test,
            "n_unique_valid": len(valid_answers),
            "n_unique_valid_raw": len(raw_valid),
            "n_unique_valid_novel": len(valid_novel_answers),
        }
        out.update({
            "test_novel_valid": out["test_unseen"],
            "creativity": len(valid_novel_answers) / n_test,
            "creativity_raw": len(raw_novel) / n_test,
            "diversity_raw": len(raw_valid) / n_test,
            "train_seen": mean("train_seen"),
            "train_validity": mean("train_validity"),
            "n_test_gen": len(acc["test_seen"]),
            "n_train_gen": len(acc["train_memorization"]),
        })
        if acc["memo_own"]:
            out["memo_own"] = mean("memo_own")
        return out


def answer_of(s):
    """答案部分：<q> 之后、首个 </a> 之前（无 <q> 时取整串）。"""
    s = s.split("<q>", 1)[1] if "<q>" in s else s
    return s.split("</a>")[0]


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
    # 事件 #4 回归：</a> 之后的批量 pad 不得改变 seen 判定
    chk("circle seen with trailing pad",
        ev.seen("zz<q><a_5><a_6><sep><a_6><a_9></a><|endoftext|><|endoftext|>"), 1)

    # sibling
    tr_s = [{"target_text": "h<q><b1_1><b2_2><a_0></a>"},
            {"target_text": "h<q><b1_5><b2_9><a_3></a>"}]
    evs = Evaluator("sibling", tr_s)
    chk("sibling valid known edges", evs.validity("q<q><b1_1><b2_2><a_0></a>"), 1)
    chk("sibling invalid wrong parent", evs.validity("q<q><b1_1><b2_2><a_3></a>"), 0)
    chk("sibling unknown node", evs.validity("q<q><b1_99><b2_2><a_0></a>"), 0)
    chk("sibling seen exact", evs.seen("q<q><b1_5><b2_9><a_3></a>"), 1)
    chk("sibling unseen valid", evs.seen("q<q><b1_5><b2_2><a_3></a>"), 0)
    chk("sibling seen with trailing pad",
        evs.seen("q<q><b1_5><b2_9><a_3></a><|endoftext|>"), 1)

    # memo_own 只比答案部分：自选配对目标来自别的 seed（hash 前缀不同）也必须判为复现（事件 #5 回归）
    other = "h2<q><b1_5><b2_9><a_3></a>"
    items = [{"type": "train", "model_output": "h1<q><b1_5><b2_9><a_3></a>",
              "target_text": "h1<q><b1_1><b2_2><a_0></a>", "own_target_text": other}]
    sc = evs.eval_items(items)
    chk("memo_own ignores hash prefix", (sc["memo_own"], sc["train_memorization"]), (1.0, 0.0))
    # creativity = 去重有效新结构 / 测试生成数
    ev2 = Evaluator("sibling", tr_s + [{"target_text": "h<q><b1_6><b2_8><a_3></a>"}])
    tests = [{"type": "test", "model_output": f"x{k}<q><b1_5><b2_8><a_3></a>", "target_text": ""}
             for k in range(3)] + [{"type": "test", "model_output": "y<q><b1_99></a>", "target_text": ""}]
    sc = ev2.eval_items(tests)
    chk("creativity counts unique novel valid / n", (sc["creativity"], sc["test_novel_valid"]),
        (0.25, 0.75))
    # Line Construction（移植自原 eval_qa，自检）
    nodes = [f"<a_{i}>" for i in range(10)]
    edges = [nodes[i] + nodes[i + 1] for i in range(9)]
    lp = "h<q>" + "<sep>".join(edges) + "</a>"
    chk("line valid path(10节点9边)", line_validity(lp), 1)
    sh = edges[:]
    import random as _r
    _r.shuffle(sh)
    chk("line valid shuffled", line_validity("x<q>" + "<sep>".join(sh) + "</a>"), 1)
    # 环（度全 2）对 line 无效
    n5 = [f"<a_{i}>" for i in range(5)]
    cyc = [n5[i] + n5[(i + 1) % 5] for i in range(5)]
    chk("line invalid cycle", line_validity("y<q>" + "<sep>".join(cyc) + "</a>"), 0)
    # 分离路径（度对但不连通）
    p1 = [f"<a_{i}>" for i in range(4)]
    p2 = [f"<a_{i}>" for i in range(4, 8)]
    two = [p1[i] + p1[i + 1] for i in range(3)] + [p2[i] + p2[i + 1] for i in range(3)]
    chk("line invalid disconnected", line_validity("z<q>" + "<sep>".join(two) + "</a>"), 0)
    # 度违规（三分叉）
    chk("line invalid degree", line_validity("w<q>" + "<sep>".join(edges[:8] + [edges[0]]) + "</a>"), 0)
    # seen：同结构重标定
    tr_l = [{"target_text": "h<q>" + "<sep>".join(edges) + "</a>"}]
    evl = Evaluator("line", tr_l)
    chk("line seen different structure", evl.seen("q<q>" + "<sep>".join([f"<a_{i+2}>" + f"<a_{i+9}>" for i in range(9)]) + "</a>"), 0)
    # 同边序＋节点重命名 → 同 canonical（与 circle 的规范化性质一致；洗边序的同结构不同码，属已知性质）
    renamed = [f"<b_{i}>" for i in range(10)]
    chk("line seen same order renamed", evl.seen("q<q>" + "<sep>".join([renamed[i] + renamed[i + 1] for i in range(9)]) + "</a>"), 1)
    chk("line seen shuffled not seen", evl.seen("q<q>" + "<sep>".join(sh) + "</a>"), 0)

    print("SELFTEST", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["circle", "sibling", "line"], required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--sibling_graph_dir", default=None,
                    help="完整 Sibling graph JSON 目录；task=sibling 时必填")
    ap.add_argument("--pred", required=True, help="预测 json（[{type,model_output,target_text}]）")
    args = ap.parse_args()
    if args.task == "sibling" and not args.sibling_graph_dir:
        ap.error("task=sibling requires --sibling_graph_dir with both complete graph JSON files")
    if args.task != "sibling" and args.sibling_graph_dir:
        ap.error("--sibling_graph_dir is only valid with task=sibling")
    with open(os.path.join(args.data_dir, "train.json"), encoding="utf-8") as f:
        tr = json.load(f)
    with open(args.pred, encoding="utf-8") as f:
        items = json.load(f)
    graph = load_sibling_graph(args.sibling_graph_dir) if args.task == "sibling" else None
    ev = Evaluator(args.task, tr, sibling_graph=graph)
    print(json.dumps(ev.eval_items(items), indent=2))
