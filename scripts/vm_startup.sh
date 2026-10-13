#!/bin/bash
# GCE startup script (runs as root on EVERY boot of the training VM). After an unplanned restart (e.g. host
# maintenance) it relaunches whatever long job was interrupted; each job resumes from its own saved state.
#   1. the pretraining run, unless it already finished successfully
#   2. the benchmark queue, if it was started and has not finished
# Set DRY_RUN=1 to only print the decision.
U=aditya.jha
H=/home/$U/tinysql
BUCKET=tinysql-data-474078649724

if [ ! -f "$H/scripts/run_main.sh" ]; then echo "startup: no code on this disk, nothing to do"; exit 0; fi
if pgrep -f "train.pretrain|benchmark_queue|run_benchmark|run_main" > /dev/null; then echo "startup: a job is already running"; exit 0; fi

if ! grep -q "training process exited with status 0" "$H/logs/main.log" 2> /dev/null; then
  [ "$DRY_RUN" = "1" ] && { echo "startup: WOULD relaunch the pretraining run"; exit 0; }
  echo "startup: relaunching the interrupted pretraining run at $(date -u +%FT%TZ)" >> "$H/logs/main.log"
  sleep 45
  sudo -u $U bash -c "cd $H && setsid nohup bash scripts/run_main.sh $BUCKET 1.2e-3 peak_tflops=121 >> /tmp/main_launcher.log 2>&1 < /dev/null &"
  exit 0
fi

if [ -f "$H/benchmark/state.json" ] && ! grep -q '"finished"' "$H/benchmark/state.json"; then
  [ "$DRY_RUN" = "1" ] && { echo "startup: WOULD resume the benchmark queue"; exit 0; }
  echo "startup: resuming the benchmark queue at $(date -u +%FT%TZ)" >> "$H/logs/benchmark.log"
  sleep 45
  sudo -u $U bash -c "cd $H && setsid nohup bash scripts/run_benchmark.sh $BUCKET >> /tmp/benchmark_launcher.log 2>&1 < /dev/null &"
  exit 0
fi
echo "startup: nothing to resume"
