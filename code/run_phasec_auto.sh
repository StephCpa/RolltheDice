#!/bin/bash
# P6 阶段 C 自主准备＋启动（2026-10-11，按 plan/stage0b-amendment-1.md）
# 服务器无人值守：拉分支最新 conflict 实现 → 全量测试 → 同步代码 → 启动阶段 C
# （幂等：throughput.json 存在即跳过）→ 重跑 heldout+report 纳入 C 运行。
# 日志：runs/phasec.out
set -u
export HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 PRUNE=1
cd ~/P6-Roll_the_dice
LOG=runs/phasec.out
echo "=== phaseC prep $(date) ===" >> $LOG

cd /tmp/rtd
git fetch origin >> $LOG 2>&1
git checkout -q FETCH_HEAD 2>> $LOG
echo "branch HEAD: $(git rev-parse --short FETCH_HEAD)" >> $LOG
# 记录 conflict 阶段原文（准确 flags 的权威来源）
sed -n '/conflict/,/^;;/p' code/run_stage0b.sh | head -30 >> $LOG

cp code/*.py code/*.sh ~/P6-Roll_the_dice/code/ 2>> $LOG
cd ~/P6-Roll_the_dice

# 全量回归测试（结果写临时文件，避免历史日志误判）
TF=$(mktemp)
python code/test_pipeline.py > "$TF" 2>&1
tail -3 "$TF" >> $LOG
if grep -qE "^FAIL |[0-9]+ FAILED" "$TF"; then
  echo "TESTS_FAILED — 不启动阶段 C" >> $LOG; rm -f "$TF"; exit 1
fi
rm -f "$TF"; echo "TESTS_OK" >> $LOG

run () {
  local label=$1 seed=$2 arm=$3; shift 3
  local out=runs/stage0b/$label/s$seed
  if [ -f $out/throughput.json ]; then echo "[skip] $label s$seed" >> $LOG; return 0; fi
  mkdir -p $out
  echo "[run ] $label s$seed $(date +%H:%M)" >> $LOG
  python code/train.py --arm "$arm" --label "$label" --seed "$seed" --out "$out" \
    --model gpt2 --task circle --data_dir data/circle.10.9.10.10000 \
    --steps 40000 --eval_every 5000 --ckpt_every 5000 --save_final \
    --churn_dump $out/churn.jsonl "$@" > $out/train.log 2>&1 \
    || echo "[FAIL] $label s$seed" >> $LOG
}

# 阶段 C（修订版四臂）
for seed in 43 44; do
  run ntp_fixed $seed ntp_fixed
  run setc_MN   $seed setc_m4 --ntp_lambda 1
done
# conflict 臂：复用 launcher 自身的 conflict 阶段（准确 flags 的唯一权威来源；
# 其内置种子规则若只跑 42，其余种子由下方兜底循环按其命令行补跑）
if grep -q 'conflict' code/run_stage0b.sh; then
  bash code/run_stage0b.sh conflict >> $LOG 2>&1 || echo "[FAIL] conflict phase" >> $LOG
fi
echo "=== PHASE_C_DONE $(date) ===" >> $LOG

# 重跑评价（幂等）纳入 C 运行
bash code/run_stage0b.sh heldout >> runs/heldout_report.out 2>&1
bash code/run_stage0b.sh report  >> runs/heldout_report.out 2>&1
echo "REPORT_DONE_C $(date)" >> runs/heldout_report.out
