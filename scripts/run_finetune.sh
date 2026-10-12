#!/bin/bash
# WikiSQL fine-tuning of the pretrained base, unattended, on the GPU VM.
#   usage: bash scripts/run_finetune.sh <bucket-name> <base-checkpoint> [extra --set key=value ...]
# Expects /tmp/wikisql.tar.bz2 (the official WikiSQL archive). Fine-tunes, scores the best epoch on the full dev set,
# copies log, metrics, predictions and the best weights to the bucket, then shuts the VM down (also on failure).
cd ~/tinysql || exit 1
export PYTHONPATH=~/tinysql PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
trap 'sudo shutdown -h now' EXIT
BUCKET=$1; BASE_CKPT=$2; shift 2
OUT=runs/finetune_wikisql
LOG=logs/finetune.log
mkdir -p logs data/raw/wikisql/extracted "$OUT"

sync_up() {
  gcloud storage cp "$LOG" "gs://$BUCKET/results/finetune/train.log" > /dev/null 2>&1
  [ -f "$OUT/metrics.jsonl" ] && gcloud storage cp "$OUT/metrics.jsonl" "gs://$BUCKET/results/finetune/metrics.jsonl" > /dev/null 2>&1
}
( while true; do sleep 120; sync_up; done ) &
SYNC_PID=$!

if [ ! -f data/raw/wikisql/extracted/data/dev.db ]; then
  tar xjf /tmp/wikisql.tar.bz2 --no-same-owner -C data/raw/wikisql/extracted >> "$LOG" 2>&1
fi
python3 -m data.wikisql_to_sql >> "$LOG" 2>&1
# run the relevant tests on THIS machine first (its Python/library versions can differ from the laptop's)
if ! python3 -m pytest -q tests/test_wikisql_converter.py tests/test_finetune.py tests/test_model.py >> "$LOG" 2>&1; then
  echo "pre-flight tests FAILED on the VM; not training" >> "$LOG"
  kill $SYNC_PID 2> /dev/null; sync_up
  exit 1
fi
echo "pre-flight tests passed" >> "$LOG"
python3 -m train.finetune_sql --config configs/finetune_wikisql.yaml --set base_ckpt="$BASE_CKPT" out_dir="$OUT" "$@" >> "$LOG" 2>&1
echo "finetune exited with status $?" >> "$LOG"
python3 -m eval.run_eval --ckpt "$OUT/best.pt" --split dev --out "$OUT" >> "$LOG" 2>&1
echo "full dev evaluation exited with status $?" >> "$LOG"

kill $SYNC_PID 2> /dev/null
sync_up
gcloud storage cp "$OUT"/dev_metrics.json "$OUT"/dev_predictions.jsonl "gs://$BUCKET/results/finetune/" > /dev/null 2>&1
gcloud storage cp "$OUT/best.pt" "gs://$BUCKET/checkpoints/finetune/best.pt" > /dev/null 2>&1
echo "all uploads done at $(date -u +%FT%TZ)" >> "$LOG"
gcloud storage cp "$LOG" "gs://$BUCKET/results/finetune/train.log" > /dev/null 2>&1
