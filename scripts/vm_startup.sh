#!/bin/bash
# GCE startup script (runs as root on EVERY boot of the training VM).
# If the full run was interrupted (host maintenance, a restart) and has not finished successfully,
# relaunch it; scripts/run_main.sh resumes from the newest checkpoint. Set DRY_RUN=1 to only print the decision.
U=aditya.jha
H=/home/$U/tinysql
BUCKET=tinysql-data-474078649724
LOG=$H/logs/main.log

if [ ! -f "$H/scripts/run_main.sh" ]; then echo "startup: no run_main.sh, nothing to do"; exit 0; fi
if grep -q "training process exited with status 0" "$LOG" 2> /dev/null; then echo "startup: run already finished, nothing to do"; exit 0; fi
if pgrep -f "train.pretrain" > /dev/null; then echo "startup: training already running, nothing to do"; exit 0; fi
if [ "$DRY_RUN" = "1" ]; then echo "startup: WOULD relaunch the training run"; exit 0; fi

echo "startup: relaunching the interrupted run at $(date -u +%FT%TZ)" >> "$LOG"
sleep 45                                            # let the NVIDIA driver finish loading
sudo -u $U bash -c "cd $H && setsid nohup bash scripts/run_main.sh $BUCKET 1.2e-3 peak_tflops=121 >> /tmp/main_launcher.log 2>&1 < /dev/null &"
