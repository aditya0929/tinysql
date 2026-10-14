#!/bin/bash
# Run the publication benchmark queue unattended on the GPU VM; shuts the VM down at the end (also on failure).
#   usage: bash scripts/run_benchmark.sh <bucket-name> [--redo key1,key2]
# Resumable: scripts/benchmark_queue.py skips every step recorded in benchmark/state.json.
cd ~/tinysql || exit 1
export PYTHONPATH=~/tinysql PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
trap 'sudo shutdown -h now' EXIT
BUCKET=$1
LOG=logs/benchmark.log
mkdir -p logs benchmark
echo "=== run_benchmark.sh started $(date -u +%FT%TZ)" >> "$LOG"

sudo python3 -m pip uninstall -y -q torchaudio >> "$LOG" 2>&1          # the image's torchaudio breaks transformers
pip install --quiet --user "transformers>=4.45" records babel >> "$LOG" 2>&1
[ -f data/raw/wikisql/extracted/data/dev.db ] || tar xjf /tmp/wikisql.tar.bz2 --no-same-owner -C data/raw/wikisql/extracted >> "$LOG" 2>&1
[ -f data/wikisql_sft/train.jsonl ] || python3 -m data.wikisql_to_sql >> "$LOG" 2>&1

# pre-flight: the WHOLE test suite on this machine's Python and libraries
if ! python3 -m pytest -q tests >> "$LOG" 2>&1; then
  echo "pre-flight FAILED on the VM; not running the queue" >> "$LOG"
  gcloud storage cp "$LOG" "gs://$BUCKET/results/benchmark/" > /dev/null 2>&1
  exit 1
fi
echo "pre-flight passed" >> "$LOG"
python3 scripts/benchmark_queue.py --bucket "$BUCKET" "${@:2}" >> "$LOG" 2>&1
echo "=== run_benchmark.sh finished $(date -u +%FT%TZ)" >> "$LOG"
gcloud storage cp "$LOG" benchmark/state.json "gs://$BUCKET/results/benchmark/" > /dev/null 2>&1
