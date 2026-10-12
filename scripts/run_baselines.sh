#!/bin/bash
# Fine-tune the SmolLM2 baselines with our exact WikiSQL recipe, unattended on the GPU VM.
#   usage: bash scripts/run_baselines.sh <bucket-name> [model ...]     (default: SmolLM2-135M then SmolLM2-360M)
# Results go to gs://<bucket>/results/baselines/<name>/ and weights to gs://<bucket>/checkpoints/baselines/<name>/.
# The VM shuts itself down at the end, also on failure.
cd ~/tinysql || exit 1
export PYTHONPATH=~/tinysql PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
trap 'sudo shutdown -h now' EXIT
BUCKET=$1; shift
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M)
LOG=logs/baselines.log
mkdir -p logs

sync_up() { gcloud storage cp "$LOG" "gs://$BUCKET/results/baselines/train.log" > /dev/null 2>&1; }
( while true; do sleep 120; sync_up; done ) &
SYNC_PID=$!

# the GCP PyTorch image ships a torchaudio built against a different torch; transformers imports it when present
# and crashes. Audio is not used here, so remove it.
sudo python3 -m pip uninstall -y -q torchaudio >> "$LOG" 2>&1
pip install --quiet --user "transformers>=4.45" >> "$LOG" 2>&1
[ -f data/raw/wikisql/extracted/data/dev.db ] || tar xjf /tmp/wikisql.tar.bz2 --no-same-owner -C data/raw/wikisql/extracted >> "$LOG" 2>&1
[ -f data/wikisql_sft/train.jsonl ] || python3 -m data.wikisql_to_sql >> "$LOG" 2>&1

# pre-flight on THIS machine: our scorer tests plus a tiny end-to-end baseline run
if ! python3 -m pytest -q tests/test_wikisql_converter.py tests/test_finetune.py tests/test_baselines.py >> "$LOG" 2>&1 || \
   ! python3 -m eval.baselines.finetune_hf --model "${MODELS[0]}" --out runs/baseline_smoke --epochs 1 --batch_size 8 \
        --limit_train 32 --eval_examples 5 --warmup_steps 2 >> "$LOG" 2>&1; then
  echo "pre-flight FAILED on the VM; not training" >> "$LOG"; kill $SYNC_PID 2> /dev/null; sync_up; exit 1
fi
echo "pre-flight passed" >> "$LOG"

for M in "${MODELS[@]}"; do
  NAME=$(basename "$M" | tr '[:upper:]' '[:lower:]')
  OUT=runs/baseline_$NAME
  echo "=== $M ($(date -u +%FT%TZ))" >> "$LOG"
  python3 -m eval.baselines.finetune_hf --model "$M" --out "$OUT" --full_dev >> "$LOG" 2>&1
  echo "$M exited with status $?" >> "$LOG"
  gcloud storage cp "$OUT"/metrics.jsonl "$OUT"/dev_greedy_metrics.json "$OUT"/dev_greedy_predictions.jsonl "gs://$BUCKET/results/baselines/$NAME/" > /dev/null 2>&1
  gcloud storage cp -r "$OUT/best" "gs://$BUCKET/checkpoints/baselines/$NAME/" > /dev/null 2>&1
  sync_up
done
kill $SYNC_PID 2> /dev/null
echo "all baselines done at $(date -u +%FT%TZ)" >> "$LOG"
sync_up
