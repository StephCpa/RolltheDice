#!/usr/bin/env bash
# P6 阶段 0 启动器：按 stage0-plan.md WP1/WP2 顺序跑七条臂（五条原臂＋两条等算力对照）。
# 用法: bash run_stage0.sh <task:circle|sibling> <steps> [seed=42]
#   环境变量: PY（解释器，默认 env312 本地路径）、EXTRA（附加给 train.py 的参数，
#   如 EXTRA="--n_probes 2048 --random_tiebreak"）
# 输出: ../runs/<日期>-<task>-s<seed>/<arm>/（只新增，不覆盖；已存在则跳过该臂）
set -u
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
TASK="${1:-circle}"
STEPS="${2:-10000}"
SEED="${3:-42}"
EXTRA="${EXTRA:-}"
DATE=$(date +%Y%m%d-%H%M)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$ROOT/env312/Scripts/python}"
DATA="$ROOT/data"
RUNS="$ROOT/runs/$DATE-$TASK-s$SEED"
mkdir -p "$RUNS"

run_arm () {
  local arm="$1" data_dir="$2" extra="${3:-}"
  local out="$RUNS/$arm"
  if [ -d "$out" ] && [ -f "$out/throughput.json" ]; then
    echo "[skip] $out 已完成"; return
  fi
  mkdir -p "$out"
  echo "[run ] $arm -> $out"
  "$PY" "$ROOT/code/train.py" --arm "$arm" --task "$TASK" \
    --data_dir "$data_dir" --out "$out" --steps "$STEPS" --seed "$SEED" \
    --save_final $extra $EXTRA 2>&1 | tee "$out/train.log" | grep -E "^\[eval\]|^step (4000|5000|10000)|throughput" 
}

if [ "$TASK" = "circle" ]; then
  HASHD="$DATA/circle.10.9.10.10000"
  NOHASHD="$DATA/circle.10.9.0.10000"
  HYBD="$DATA/circle_hybrid.10.9.10.10000"
else
  HASHD="$DATA/sibling.10.1000.10.50000"
  NOHASHD="$DATA/sibling.10.1000.0.50000"
  HYBD="$DATA/sibling_hybrid.10.1000.10.50000"
fi

run_arm ntp_noseed  "$NOHASHD"
run_arm ntp_fixed   "$HASHD"
run_arm ntp_randperm "$HASHD"
run_arm tl_hybrid   "$HYBD"
run_arm setc_m4     "$HASHD" "--churn_dump $RUNS/setc_m4/churn.jsonl"
# 等算力对照：与 setc 同目标/同算力/同批组织，仅配对选择规则不同（恒等 / 随机）
run_arm tl_fixed_m4    "$HASHD" "--churn_dump $RUNS/tl_fixed_m4/churn.jsonl"
run_arm tl_randperm_m4 "$HASHD" "--churn_dump $RUNS/tl_randperm_m4/churn.jsonl"
echo "ALL DONE -> $RUNS"
