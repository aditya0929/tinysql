#!/bin/bash
# The full 125M pretraining run: unattended, resumable, backed up to the bucket.
#   usage: bash scripts/run_main.sh <bucket-name> <peak_lr> [extra --set key=value ...]
#
# - resumes from the newest checkpoint on this disk, or else from the bucket (so a replaced VM continues the run)
# - every 5 minutes syncs checkpoints, metrics and the log to gs://<bucket>/checkpoints/main and /results/main
# - when training ends (success, crash, or kill) the VM shuts itself down, so a forgotten VM cannot keep billing
cd ~/tinysql || exit 1
export PYTHONPATH=~/tinysql PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
trap 'sudo shutdown -h now' EXIT
BUCKET=$1; LR=$2; shift 2
OUT=runs/main
mkdir -p logs "$OUT"

if ! ls "$OUT"/ckpt_*.pt > /dev/null 2>&1; then
  latest=$(gcloud storage ls "gs://$BUCKET/checkpoints/main/ckpt_*.pt" 2> /dev/null | sort | tail -1)
  if [ -n "$latest" ]; then
    echo "restoring $latest" >> logs/main.log
    gcloud storage cp "$latest" "$OUT/" >> logs/main.log 2>&1
  fi
fi

sync_up() {
  gcloud storage rsync "$OUT" "gs://$BUCKET/checkpoints/main" --delete-unmatched-destination-objects --exclude=".*tb.*" > /dev/null 2>&1
  gcloud storage cp logs/main.log "gs://$BUCKET/results/main/train.log" > /dev/null 2>&1
  gcloud storage cp "$OUT/metrics.jsonl" "gs://$BUCKET/results/main/metrics.jsonl" > /dev/null 2>&1
}
( while true; do sleep 300; sync_up; done ) &
SYNC_PID=$!

python3 -m train.pretrain --config configs/tinysql_125m.yaml --set peak_lr="$LR" out_dir="$OUT" tensorboard=false "$@" >> logs/main.log 2>&1
STATUS=$?
kill $SYNC_PID 2> /dev/null
sync_up
echo "training process exited with status $STATUS at $(date -u +%FT%TZ)" >> logs/main.log
gcloud storage cp logs/main.log "gs://$BUCKET/results/main/train.log" > /dev/null 2>&1
