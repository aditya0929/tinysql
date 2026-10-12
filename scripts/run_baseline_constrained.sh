#!/bin/bash
# Score the fine-tuned SmolLM2 baselines with OUR constrained decoder (schema, and schema + values), on the GPU VM.
#   usage: bash scripts/run_baseline_constrained.sh <bucket-name>
# Uses the fine-tuned weights left on the VM disk by run_baselines.sh (falls back to the copies in the bucket).
cd ~/tinysql || exit 1
export PYTHONPATH=~/tinysql PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
trap 'sudo shutdown -h now' EXIT
BUCKET=$1
LOG=logs/baseline_constrained.log
mkdir -p logs
sync_up() { gcloud storage cp "$LOG" "gs://$BUCKET/results/baselines/constrained.log" > /dev/null 2>&1; }
( while true; do sleep 120; sync_up; done ) &
SYNC_PID=$!

sudo python3 -m pip uninstall -y -q torchaudio >> "$LOG" 2>&1
pip install --quiet --user "transformers>=4.45" >> "$LOG" 2>&1
if ! python3 -m pytest -q tests/test_baselines.py tests/test_constrained.py >> "$LOG" 2>&1; then
  echo "pre-flight FAILED on the VM; not running" >> "$LOG"; kill $SYNC_PID 2> /dev/null; sync_up; exit 1
fi
echo "pre-flight passed" >> "$LOG"

for NAME in smollm2-135m smollm2-360m; do
  DIR=runs/baseline_$NAME/best
  [ -f "$DIR/config.json" ] || { mkdir -p "$DIR"; gcloud storage cp -r "gs://$BUCKET/checkpoints/baselines/$NAME/best/*" "$DIR/" >> "$LOG" 2>&1; }
  for DECODE in schema values; do
    python3 -m eval.baselines.constrained_hf --model_dir "$DIR" --decode "$DECODE" --out "runs/baseline_$NAME/constrained" >> "$LOG" 2>&1
    echo "$NAME $DECODE exited with status $?" >> "$LOG"
    gcloud storage cp runs/baseline_$NAME/constrained/dev_${DECODE}_* "gs://$BUCKET/results/baselines/$NAME/" > /dev/null 2>&1
  done
done
kill $SYNC_PID 2> /dev/null
echo "all constrained baseline evaluations done at $(date -u +%FT%TZ)" >> "$LOG"
sync_up
