"""Compare the learning-rate sweep runs stored in the bucket.

    python scripts/compare_sweep.py            # downloads results/sweep_lr_*/metrics.jsonl and prints a table

The score used to rank runs is the final validation loss averaged with the weights of the overall training mix
(78% FineWeb-Edu, 18% SQL code, 3.3% Stack Exchange, 0.7% Gretel), since that is what the full run optimises.
"""
import glob
import json
import os
import subprocess
import sys

BUCKET = "gs://tinysql-data-474078649724/results"
GCLOUD = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Cloud SDK", "google-cloud-sdk", "bin", "gcloud.cmd")
LOCAL = os.path.join("runs_local", "sweep")
MIX = {"fineweb_edu": 0.775, "stack_sql": 0.18, "stackexchange": 0.036, "gretel_sql": 0.009}


def download():
    os.makedirs(LOCAL, exist_ok=True)
    gcloud = GCLOUD if os.path.exists(GCLOUD) else "gcloud"
    subprocess.run([gcloud, "storage", "cp", "-r", BUCKET + "/sweep_lr_*", LOCAL], check=True, capture_output=True)


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def summarize(rows):
    train = [r for r in rows if "loss" in r]
    vals = [r for r in rows if any(k.startswith("val_loss/") for k in r)]
    last_val = {k.split("/", 1)[1]: v for k, v in vals[-1].items() if k.startswith("val_loss/")} if vals else {}
    tail = [r["loss"] for r in train[-8:]]
    return {
        "steps": train[-1]["step"] if train else 0,
        "train_loss_tail": sum(tail) / len(tail) if tail else float("nan"),
        "val": last_val,
        "max_grad_norm": max((r["grad_norm"] for r in train), default=float("nan")),
        "skipped": train[-1].get("skipped_steps", 0) if train else 0,
        "tok_s": sorted(r["tokens_per_s"] for r in train)[len(train) // 2] if train else 0,
    }


def score(val):
    return sum(MIX[s] * val[s] for s in MIX if s in val) / sum(MIX[s] for s in MIX if s in val)


def main():
    if "--no-download" not in sys.argv:
        download()
    runs = {}
    for d in glob.glob(os.path.join(LOCAL, "sweep_lr_*")):
        lr = float(os.path.basename(d).split("sweep_lr_")[1])
        path = os.path.join(d, "metrics.jsonl")
        if os.path.exists(path):
            runs[lr] = summarize(load(path))
    print(f"{'peak LR':>9} {'steps':>6} {'train loss':>10} {'fineweb':>8} {'sql':>7} {'stackex':>8} {'gretel':>7} {'weighted':>9} {'max |g|':>8} {'skipped':>8}")
    best = None
    for lr in sorted(runs):
        r = runs[lr]
        v = r["val"]
        sc = score(v) if v else float("nan")
        if v and (best is None or sc < best[1]):
            best = (lr, sc)
        print(f"{lr:>9.1e} {r['steps']:>6} {r['train_loss_tail']:>10.3f} {v.get('fineweb_edu', float('nan')):>8.3f} {v.get('stack_sql', float('nan')):>7.3f} "
              f"{v.get('stackexchange', float('nan')):>8.3f} {v.get('gretel_sql', float('nan')):>7.3f} {sc:>9.3f} {r['max_grad_norm']:>8.2f} {r['skipped']:>8}")
    if best:
        edge = best[0] in (min(runs), max(runs))
        print(f"\nbest by weighted validation loss: {best[0]:.1e}" + ("   (at the edge of the sweep: extend it before trusting)" if edge else ""))


if __name__ == "__main__":
    main()
