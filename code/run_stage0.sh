#!/usr/bin/env bash
# P6 阶段 0 启动器：按 stage0-plan.md WP1/WP2 顺序跑五条臂。
# 用法: bash run_stage0.sh <task:circle|sibling> <steps>
# 输出: ../runs/<日期>/<task>-<arm>/（只新增，不覆盖；已存在则跳过该臂）
set -u
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
TASK="${1:-circle}"
STEPS="${2:-10000}"
DATE=$(date +%Y%m%d-%H%M)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/env312/Scripts/python"
DATA="$ROOT/data"
RUNS="$ROOT/runs/$DATE-$TASK"
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
    --data_dir "$data_dir" --out "$out" --steps "$STEPS" \
    --save_final $extra 2>&1 | tee "$out/train.log" | grep -E "^\[eval\]|^step (4000|5000|10000)|throughput" 
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
echo "ALL DONE -> $RUNS"
