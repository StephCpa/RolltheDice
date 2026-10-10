#!/usr/bin/env bash
# P6 阶段 0b 启动器（plan/stage0b-plan.md）。用法：bash code/run_stage0b.sh <阶段>
#   A          已有末模型重评（memo_own/留出集/无 seed 采样）、同口径间隔基线、C01、tl_hybrid 保真度——不训练
#   B          种子 42：ntp_fixed, ntp_fixed_grp, tl_NN, setc_MM（含分支状态）, setc_MN, setc_NM
#   lock       冻结-锁定等价检验（从 setc_MM 的 state_step$LOCK_AT 分两支续训 $LOCK_LEN 步并逐位比较）
#   intervene  晚期干预：从 state_step$INTERVENE_AT 以 --ntp_lambda 1 续训到 $STEPS
#   C          种子 $SEEDS_C：ntp_fixed, setc_MN, setc_MM, tl_NN；setc_M05 ×（42 + $SEEDS_C）
#   replay     π* 回放（各种子用自己 setc_MM 的 final_pairing.json）：tl_NN_rep, ntp_fixed_rep
#   diag       TL/setc 运行：churn（结构级）+ 逐检查点 gap
#   heldout    生成留出集（若无）→ 开发集选检查点 → 留出集一次性评测（PRUNE=1 删未选检查点）
#   report     汇总（参照 ntp_fixed，margin 0.03，min_seeds 3）
# 环境变量（括号内为缺省）：PY(python) DATA(data/circle.10.9.10.10000) NOHASH(data/circle.10.9.0.10000)
#   OUT(runs/stage0b) SRV(runs/server-circle-40k，阶段 A 的已有运行) MODEL(gpt2) STEPS(40000) EVERY(5000)
#   STATE_AT(10000,14000,20000) LOCK_AT(14000) LOCK_LEN(2000) INTERVENE_AT(20000) SEEDS_C("43 44")
#   HELD_N(2048) GPUS（如 "0 1 2 3"：同一阶段的运行按批轮转并行）EXTRA（附加给 train.py 的参数）
set -euo pipefail
PHASE="${1:?见脚本头部用法}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"
DATA="${DATA:-$ROOT/data/circle.10.9.10.10000}"
NOHASH="${NOHASH:-$ROOT/data/circle.10.9.0.10000}"
OUT="${OUT:-$ROOT/runs/stage0b}"
SRV="${SRV:-$ROOT/runs/server-circle-40k}"
MODEL="${MODEL:-gpt2}"
STEPS="${STEPS:-40000}"
EVERY="${EVERY:-5000}"
STATE_AT="${STATE_AT:-10000,14000,20000}"
LOCK_AT="${LOCK_AT:-14000}"
LOCK_LEN="${LOCK_LEN:-2000}"
INTERVENE_AT="${INTERVENE_AT:-20000}"
SEEDS_C="${SEEDS_C:-43 44}"
EXTRA="${EXTRA:-}"
HELD_N="${HELD_N:-2048}"
read -r -a GPU_LIST <<< "${GPUS:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
C="$ROOT/code"
HELD="$DATA/heldout.json"
JOBS=0

launch () {  # 无 GPUS 时前台顺序执行；否则按 GPU 轮转后台执行，每满一批等待
  if [ ${#GPU_LIST[@]} -eq 0 ]; then "$@"; return; fi
  local g=${GPU_LIST[$((JOBS % ${#GPU_LIST[@]}))]}
  CUDA_VISIBLE_DEVICES=$g "$@" &
  JOBS=$((JOBS + 1))
  if [ $((JOBS % ${#GPU_LIST[@]})) -eq 0 ]; then wait; fi
}

train () {  # train <label> <seed> <arm> [train.py 参数...]
  local label=$1 seed=$2 arm=$3; shift 3
  local out="$OUT/$label/s$seed"
  if [ -f "$out/throughput.json" ]; then echo "[skip] $out"; return 0; fi
  mkdir -p "$out"
  echo "[run ] $label s$seed"
  # shellcheck disable=SC2086
  "$PY" "$C/train.py" --arm "$arm" --label "$label" --seed "$seed" --out "$out" \
    --model "$MODEL" --task circle --data_dir "$DATA" --steps "$STEPS" \
    --eval_every "$EVERY" --ckpt_every "$EVERY" --save_final \
    --churn_dump "$out/churn.jsonl" "$@" $EXTRA > "$out/train.log" 2>&1 \
    || echo "[FAIL] $out（见 train.log）"
}

branch () {  # branch <label> <state_dir> <steps> [参数...]：从完整状态续训的分支（种子 42）
  local label=$1 st=$2 steps=$3; shift 3
  local out="$OUT/$label/s42"
  if [ -f "$out/throughput.json" ]; then echo "[skip] $out"; return 0; fi
  [ -f "$st/train_state.pt" ] || { echo "[miss] $st（先跑阶段 B）"; return 0; }
  mkdir -p "$out"
  echo "[run ] $label（续训自 $st 到 $steps）"
  # shellcheck disable=SC2086
  "$PY" "$C/train.py" --arm setc_m4 --label "$label" --seed 42 --out "$out" \
    --model "$MODEL" --task circle --data_dir "$DATA" --steps "$steps" \
    --eval_every "$EVERY" --ckpt_every "$EVERY" --save_final \
    --churn_dump "$out/churn.jsonl" --resume_from "$st" "$@" $EXTRA > "$out/train.log" 2>&1 \
    || echo "[FAIL] $out（见 train.log）"
}

heldout_file () { [ -f "$HELD" ] || "$PY" "$C/make_heldout.py" --data_dir "$DATA" --n "$HELD_N"; }

try () { "$@" || echo "[FAIL] $*"; }   # 阶段 A 的各项相互独立：单项失败只记录、不中断

phase_A () {
  heldout_file
  local A="$OUT/phaseA"; mkdir -p "$A"
  for arm in ntp_fixed_v2 ntp_randperm_v2 tl_hybrid_v2 setc_m4_v2; do
    local md="$SRV/$arm/final_model"
    [ -d "$md" ] || { echo "[miss] $md"; continue; }
    local pf=()
    [ -f "$SRV/$arm/pistar_final.json" ] && pf=(--pairing_file "$SRV/$arm/pistar_final.json")
    try "$PY" "$C/rescore.py" --model_dir "$md" --data_dir "$DATA" --out "$A/$arm/dev" \
      --n_test 128 --step 40000 "${pf[@]}"
    try "$PY" "$C/rescore.py" --model_dir "$md" --data_dir "$DATA" --out "$A/$arm/heldout" \
      --probe_file "$HELD" --step 40000 "${pf[@]}"
  done
  if [ -d "$SRV/ntp_noseed_v2/final_model" ]; then
    for T in 0.5 1.0; do
      try "$PY" "$C/rescore.py" --model_dir "$SRV/ntp_noseed_v2/final_model" --data_dir "$NOHASH" \
        --out "$A/ntp_noseed_v2/T$T" --n_test 1024 --sample --temp "$T"
    done
  fi
  [ -d "$SRV/ntp_fixed_v2/final_model" ] && try "$PY" "$C/diagnose.py" --mode gap \
    --model_dir "$SRV/ntp_fixed_v2/final_model" --data_dir "$DATA" --G 256 --out "$A/gap_ntp_fixed_v2.json"
  if [ -d "$SRV/setc_m4_v2/final_model" ]; then
    try "$PY" "$C/diagnose.py" --mode gap --model_dir "$SRV/setc_m4_v2/final_model" --data_dir "$DATA" \
      --G 256 --ref_pairing "$SRV/setc_m4_v2/pistar_final.json" --out "$A/gap_setc_m4_v2.json"
    try "$PY" "$C/throughput.py" --model "$SRV/setc_m4_v2/final_model" --data_dir "$DATA" \
      --m 4,8,16 --G 16 --out "$A/c01.json"
  fi
  if [ -d "$SRV/tl_hybrid_v2/final_model" ]; then
    try "$PY" "$C/diagnose.py" --mode readout --model_dir "$SRV/tl_hybrid_v2/final_model" \
      --data_dir "$DATA" --out "$A/tl_hybrid_v2_readout.json"
    try "$PY" "$C/rescore.py" --model_dir "$SRV/tl_hybrid_v2/final_model" --data_dir "$DATA" \
      --out "$A/tl_hybrid_v2/T1.0" --n_test 128 --sample --temp 1.0
  fi
}

phase_B () {
  launch train ntp_fixed 42 ntp_fixed
  launch train ntp_fixed_grp 42 ntp_fixed_grouped
  launch train tl_NN 42 tl_fixed_m4
  launch train setc_MM 42 setc_m4 --state_at "$STATE_AT"
  launch train setc_MN 42 setc_m4 --ntp_lambda 1
  launch train setc_NM 42 setc_m4 --tl_target nominal
  wait
}

phase_lock () {
  local st="$OUT/setc_MM/s42/state_step$LOCK_AT" end=$((LOCK_AT + LOCK_LEN))
  launch branch setc_MM_online "$st" "$end"
  launch branch setc_MM_lock "$st" "$end" --lock_at "$LOCK_AT"
  wait
  "$PY" "$C/diagnose.py" --mode compare --run_a "$OUT/setc_MM_online/s42" --run_b "$OUT/setc_MM_lock/s42" \
    --churn_dump "$OUT/setc_MM_online/s42/churn.jsonl" --churn_b "$OUT/setc_MM_lock/s42/churn.jsonl" \
    --out "$OUT/lock_compare.json"
}

phase_intervene () {
  branch setc_MM_late_MN "$OUT/setc_MM/s42/state_step$INTERVENE_AT" "$STEPS" --ntp_lambda 1
}

phase_C () {
  for s in $SEEDS_C; do
    launch train ntp_fixed "$s" ntp_fixed
    launch train setc_MN "$s" setc_m4 --ntp_lambda 1
    launch train setc_MM "$s" setc_m4
    launch train tl_NN "$s" tl_fixed_m4
  done
  for s in 42 $SEEDS_C; do launch train setc_M05 "$s" setc_m4 --ntp_lambda 0.5; done
  wait
}

phase_replay () {
  for s in 42 $SEEDS_C; do
    local src="$OUT/setc_MM/s$s/final_pairing.json"
    [ -f "$src" ] || { echo "[miss] $src"; continue; }
    launch train tl_NN_rep "$s" tl_fixed_m4 --pairing_file "$src"
    launch train ntp_fixed_rep "$s" ntp_fixed --pairing_file "$src"
  done
  wait
}

phase_diag () {
  for d in "$OUT"/setc_*/s* "$OUT"/tl_NN*/s*; do
    [ -d "$d" ] || continue
    local seed=${d##*/s}
    [ -f "$d/churn.jsonl" ] && "$PY" "$C/diagnose.py" --mode churn --churn_dump "$d/churn.jsonl" \
      --data_dir "$DATA" --train_seed "$seed" --out "$d/churn_analysis.json" > /dev/null
    [ -d "$d/ckpts" ] && "$PY" "$C/diagnose.py" --mode gap --model_dir "$d/ckpts/step*" \
      --data_dir "$DATA" --train_seed "$seed" --G 128 --out "$d/gap.json"
  done
}

phase_heldout () {
  heldout_file
  "$PY" "$C/report.py" --runs "$OUT/*/s*" --emit_selection "$OUT/selection.tsv" > /dev/null
  while IFS=$'\t' read -r run step ck; do
    if [ -f "$run/heldout/scores.jsonl" ] && grep -q "\"step\": $step[,}]" "$run/heldout/scores.jsonl"; then
      echo "[skip] $run 留出集 @ $step"; continue
    fi
    echo "[held] $run @ $step"
    "$PY" "$C/rescore.py" --model_dir "$ck" --data_dir "$DATA" --probe_file "$HELD" \
      --out "$run/heldout" --step "$step" > "$run/heldout.log" 2>&1 || echo "[FAIL] $run（见 heldout.log）"
    if [ "${PRUNE:-0}" = 1 ] && [ -d "$run/ckpts" ]; then
      find "$run/ckpts" -mindepth 1 -maxdepth 1 -type d ! -name "step$step" -exec rm -rf {} +
    fi
  done < "$OUT/selection.tsv"
}

phase_report () {
  "$PY" "$C/report.py" --runs "$OUT/*/s*" --ref ntp_fixed --margin 0.03 --min_seeds 3 \
    --out "$OUT/report.md" --json "$OUT/report.json"
}

case "$PHASE" in
  A) phase_A ;; B) phase_B ;; lock) phase_lock ;; intervene) phase_intervene ;;
  C) phase_C ;; replay) phase_replay ;; diag) phase_diag ;;
  heldout) phase_heldout ;; report) phase_report ;;
  *) echo "未知阶段：$PHASE"; exit 2 ;;
esac
