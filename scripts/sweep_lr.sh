#!/bin/bash
# Learning-rate sweep on the 25M proxy model. Runs unattended on the GPU VM.
#   usage: bash scripts/sweep_lr.sh <bucket-name> <lr1> <lr2> ...
# Results go to gs://<bucket>/results/sweep_lr_<lr>/ ; the VM shuts itself down when everything is done
# (also if a run crashes), so a forgotten VM cannot keep billing.
cd ~/tinysql || exit 1
export PYTHONPATH=~/tinysql PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
trap 'sudo shutdown -h now' EXIT
BUCKET=$1; shift
mkdir -p logs
for lr in "$@"; do
  out=runs/sweep_lr_$lr
  rm -rf "$out"
  python3 -m train.pretrain --config configs/proxy_25m.yaml --set peak_lr="$lr" out_dir="$out" > "logs/sweep_lr_$lr.log" 2>&1
  gcloud storage cp "$out/metrics.jsonl" "gs://$BUCKET/results/sweep_lr_$lr/metrics.jsonl" > /dev/null 2>&1
  gcloud storage cp "logs/sweep_lr_$lr.log" "gs://$BUCKET/results/sweep_lr_$lr/train.log" > /dev/null 2>&1
  rm -f "$out"/ckpt_*.pt
done
echo "done $(date -u +%FT%TZ)" > logs/sweep.done
gcloud storage cp logs/sweep.done "gs://$BUCKET/results/sweep.done" > /dev/null 2>&1
