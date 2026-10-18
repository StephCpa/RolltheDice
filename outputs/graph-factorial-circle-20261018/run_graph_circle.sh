#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lzn/P6-Roll_the_dice}"
PY="${PY:-/home/lzn/envs/dllm_cf_l40_py312/bin/python}"
CODE="$ROOT/code-v1274bde/code"
DATA_ROOT="$ROOT/data/circle-graph-factorial-development-20261017"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/icml2027-graph-factorial-circle-20261018-v3}"
MODEL="${MODEL:-$ROOT/models/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e}"
MODEL_REVISION="gpt2-local-snapshot-607a30d7"

mkdir -p "$OUT_ROOT/logs"
declare -a PIDS=()
declare -a GPUS=(0 1)
next_gpu=0

start_one() {
  local cell="$1" seed="$2" gpu="$3"
  local data="$DATA_ROOT/$cell"
  local out="$OUT_ROOT/${cell}__seed${seed}"
  [ -d "$data" ] || { echo "missing data: $data" >&2; exit 2; }
  [ ! -e "$out" ] || { echo "refusing to overwrite: $out" >&2; exit 2; }
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" "$CODE/train.py" \
    --arm ntp_fixed --label "graph_${cell}" --data_dir "$data" --task circle \
    --out "$out" --model "$MODEL" --steps 40000 --bs 128 --warmup 2000 \
    --seed "$seed" --eval_every 5000 --ckpt_every 5000 --n_probes 256 \
    --save_final --model_revision "$MODEL_REVISION" \
    > "$OUT_ROOT/logs/${cell}__seed${seed}.log" 2>&1 < /dev/null &
  PIDS+=("$!")
  echo "started $cell seed=$seed gpu=$gpu pid=${PIDS[-1]}"
}

wait_one() {
  wait "${PIDS[0]}"
  PIDS=("${PIDS[@]:1}")
}

for cell in G12 G14 G22 fresh20k; do
  for seed in 42 43 44; do
    while [ "${#PIDS[@]}" -ge 2 ]; do wait_one; done
    start_one "$cell" "$seed" "${GPUS[$next_gpu]}"
    next_gpu=$(( (next_gpu + 1) % 2 ))
  done
done
while [ "${#PIDS[@]}" -gt 0 ]; do wait_one; done

