"""Unattended, resumable benchmark queue for the GPU VM (the publication experiments).

    python scripts/benchmark_queue.py --bucket tinysql-data-474078649724 [--dry-run]

Phases (every model gets the same treatment; selection always uses dev execution accuracy on the first 1,000 dev
questions, as in all earlier runs):
  1  learning-rate sweep: each model fine-tuned at 5e-5, 1e-4, 2e-4 (seed 0)
  2  at each model's best LR: seeds 0, 1, 2 for TinySQL and both SmolLM2 models, seed 0 for GPT-2 and Pythia;
     every final run scored on the full dev set greedily and with schema+values constraints
     (GPT-2 / Pythia also with schema-only constraints)
  3  no-pretraining control: TinySQL architecture from random weights, fine-tuned at 1e-4 and 1e-3
  4  ablations on the 25M proxy: pretrain 5 variants (150M tokens each), fine-tune each, score greedy on full dev

Every finished step is recorded in benchmark/state.json (mirrored to the bucket), so the queue resumes after an
interruption. Results go to gs://<bucket>/results/benchmark/<job>/. Constrained evaluations are CPU-bound, so up to
three run in the background while the next fine-tune trains. Weights are deleted once a run is fully scored.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

MODELS = {
    "tinysql": None,                                  # ours: train.finetune_sql from the pretrained checkpoint
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M",
    "gpt2": "openai-community/gpt2",
    "pythia-160m": "EleutherAI/pythia-160m",
}
MAIN = ["tinysql", "smollm2-135m", "smollm2-360m"]
LRS = [5e-5, 1e-4, 2e-4]
BASE_CKPT = "runs/main/ckpt_0008600.pt"
ABLATIONS = ["base", "learned_pos", "mha", "untied", "general_only"]
STATE_FILE = "benchmark/state.json"
LOG = "logs/benchmark.log"
MAX_BACKGROUND = 3


class Queue:
    def __init__(self, bucket, dry_run):
        self.bucket, self.dry = bucket, dry_run
        self.background = []
        os.makedirs("benchmark", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        self.state = json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {"done": {}, "failed": {}}

    # ------------------------------------------------------------------ plumbing
    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC] {msg}"
        print(line, flush=True)
        with open(LOG, "a") as f:
            f.write(line + "\n")

    def save(self):
        json.dump(self.state, open(STATE_FILE, "w"), indent=1)
        if not self.dry:
            subprocess.run(["gcloud", "storage", "cp", STATE_FILE, LOG, f"gs://{self.bucket}/results/benchmark/"],
                           capture_output=True)

    def done(self, key):
        return key in self.state["done"]

    def step(self, key, cmd, background_then=None):
        """Run one step unless already done. background_then: shell to run after cmd, in the background."""
        if self.done(key):
            return True
        self.log(f"START {key}: {' '.join(cmd)}")
        if self.dry:
            self.state["done"][key] = "dry-run"
            return True
        if background_then is not None:
            while len([p for p in self.background if p[1].poll() is None]) >= MAX_BACKGROUND:
                time.sleep(20)
            self.reap()
            # exit with the EVALUATION's status (not the clean-up's), and keep the weights if it failed
            shell = f"{' '.join(cmd)} >> {LOG} 2>&1; rc=$?; if [ $rc -eq 0 ]; then {background_then}; fi; exit $rc"
            self.background.append((key, subprocess.Popen(["bash", "-c", shell])))
            return True
        t0 = time.time()
        with open(LOG, "a") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            self.state["done"][key] = round(time.time() - t0)
            self.state["failed"].pop(key, None)
            self.log(f"DONE  {key} in {time.time() - t0:.0f}s")
        else:
            self.state["failed"][key] = rc
            self.log(f"FAIL  {key} (exit {rc})")
        self.save()
        return rc == 0

    def reap(self, wait=False):
        for key, p in list(self.background):
            if wait:
                p.wait()
            if p.poll() is not None:
                if p.returncode == 0:
                    self.state["done"][key] = "background"
                    self.state["failed"].pop(key, None)
                else:
                    self.state["failed"][key] = p.returncode
                self.log(f"{'DONE ' if p.returncode == 0 else 'FAIL '} {key} (background)")
                self.background.remove((key, p))
                self.save()

    def upload(self, job):
        if self.dry:
            return
        files = [f for pat in ("metrics.jsonl", "*_metrics.json", "*_predictions.jsonl")
                 for f in glob.glob(os.path.join("runs/bench", job, pat))]
        if files:
            subprocess.run(["gcloud", "storage", "cp", *files, f"gs://{self.bucket}/results/benchmark/{job}/"], capture_output=True)

    # ------------------------------------------------------------------ jobs
    @staticmethod
    def job(model, lr, seed):
        return f"{model}_lr{lr:g}_s{seed}"

    def weights(self, model, job):
        ours = model in ("tinysql", "control") or model.startswith("abl_")
        return f"runs/bench/{job}/best.pt" if ours else f"runs/bench/{job}/best"

    def preload(self):
        """Seed-0 runs at LR 1e-4 of the three main models were done earlier with this exact recipe and scored on the
        full dev set (greedy and schema+values). Their files were uploaded to the bucket; reuse them."""
        for model in MAIN:
            job = self.job(model, 1e-4, 0)
            out = f"runs/bench/{job}"
            os.makedirs(out, exist_ok=True)
            if not self.dry:
                subprocess.run(["gcloud", "storage", "cp", f"gs://{self.bucket}/results/benchmark/{job}/*", out + "/"], capture_output=True)
            for key, needed in ((f"ft:{job}", "metrics.jsonl"), (f"dev_greedy:{job}", "dev_greedy_predictions.jsonl"),
                                (f"dev_values:{job}", "dev_values_predictions.jsonl")):
                if not self.done(key) and (self.dry or os.path.exists(os.path.join(out, needed))):
                    self.state["done"][key] = "preloaded"
        self.save()

    def finetune(self, model, lr, seed, base=BASE_CKPT):
        job = self.job(model, lr, seed)
        out = f"runs/bench/{job}"
        if model in ("tinysql", "control"):
            cmd = [sys.executable, "-m", "train.finetune_sql", "--config", "configs/finetune_wikisql.yaml", "--set",
                   f"base_ckpt={base}", f"out_dir={out}", f"peak_lr={lr}", f"seed={seed}"]
        else:
            cmd = [sys.executable, "-m", "eval.baselines.finetune_hf", "--model", MODELS[model], "--out", out,
                   "--peak_lr", str(lr), "--seed", str(seed)]
        if not self.done(f"ft:{job}"):
            self.reap(wait=True)        # background evaluations hold 2-3 GB each: a fine-tune next to them ran out of memory
        ok = self.step(f"ft:{job}", cmd)
        self.upload(job)
        return ok

    def best_lr(self, model, lrs=LRS, prefix=None):
        scores = {}
        for lr in lrs:
            path = f"runs/bench/{self.job(prefix or model, lr, 0)}/metrics.jsonl"
            if os.path.exists(path):
                accs = [json.loads(l).get("dev_exec_acc") for l in open(path)]
                accs = [a for a in accs if a is not None]
                if accs:
                    scores[lr] = max(accs)
        if not scores:
            return None
        best = max(scores, key=scores.get)
        self.log(f"best LR for {prefix or model}: {best:g} (dev subset {scores})")
        self.state.setdefault("best_lr", {})[prefix or model] = {"lr": best, "scores": {str(k): v for k, v in scores.items()}}
        return best

    def score(self, model, job, modes):
        """Full-dev scoring of a finished fine-tune: greedy in the foreground, constrained in the background,
        then delete its weights once everything has finished."""
        out, w = f"runs/bench/{job}", self.weights(model, job)
        ours = model in ("tinysql", "control") or model.startswith("abl_")
        if "greedy" in modes:
            cmd = ([sys.executable, "-m", "eval.run_eval", "--ckpt", w, "--split", "dev", "--decode", "greedy", "--out", out] if ours
                   else [sys.executable, "-m", "eval.baselines.eval_hf", "--model_dir", w, "--out", out])
            self.step(f"dev_greedy:{job}", cmd)
            self.upload(job)
        constrained = [m for m in modes if m != "greedy"]
        for i, mode in enumerate(constrained):
            cmd = ([sys.executable, "-m", "eval.run_eval", "--ckpt", w, "--split", "dev", "--decode", mode, "--out", out] if ours
                   else [sys.executable, "-m", "eval.baselines.constrained_hf", "--model_dir", w, "--decode", mode, "--out", out])
            after = (f"gcloud storage cp {out}/dev_{mode}_*.json* gs://{self.bucket}/results/benchmark/{job}/ > /dev/null 2>&1"
                     + (f" ; rm -rf {w}" if i == len(constrained) - 1 else ""))
            self.step(f"dev_{mode}:{job}", cmd, background_then=after)
        if not constrained and not self.dry:
            shutil.rmtree(w, ignore_errors=True) if os.path.isdir(w) else (os.path.exists(w) and os.remove(w))

    # ------------------------------------------------------------------ phases
    def run(self, redo=()):
        """redo: step keys to run again even though they are recorded as done. Failed steps are always retried."""
        for key in redo:
            self.state["done"].pop(key, None)
        self.state.pop("finished", None)
        self.preload()
        self.log("=== phase 1: learning-rate sweep")
        for model in MODELS:
            for lr in LRS:
                self.finetune(model, lr, 0)

        self.log("=== phase 2: seeds at each model's best learning rate, full-dev scoring")
        for model in MODELS:
            lr = self.best_lr(model)
            if lr is None:
                self.log(f"no sweep result for {model}; skipping")
                continue
            for other in LRS:                                        # sweep runs not chosen: free their disk
                if other != lr and not self.dry:
                    w = self.weights(model, self.job(model, other, 0))
                    shutil.rmtree(w, ignore_errors=True) if os.path.isdir(w) else (os.path.exists(w) and os.remove(w))
            seeds = [0, 1, 2] if model in MAIN else [0]
            for seed in seeds:
                if seed and not self.finetune(model, lr, seed):
                    continue
                modes = ["greedy", "values"] if model in MAIN else ["greedy", "schema", "values"]
                self.score(model, self.job(model, lr, seed), modes)
        self.save()

        self.log("=== phase 3: no-pretraining control")
        for lr in (1e-4, 1e-3):
            self.finetune("control", lr, 0, base="random")
        lr = self.best_lr("control", lrs=(1e-4, 1e-3))
        if lr is not None:
            self.score("control", self.job("control", lr, 0), ["greedy", "values"])

        self.log("=== phase 4: architecture and data ablations on the 25M proxy")
        for name in ABLATIONS:
            pre = f"runs/ablation_{name}"
            self.step(f"pretrain:{name}", [sys.executable, "-m", "train.pretrain", "--config", f"configs/ablation_{name}.yaml",
                                           "--set", "tensorboard=false"])
            if not self.dry:
                subprocess.run(["gcloud", "storage", "cp", f"{pre}/metrics.jsonl", f"gs://{self.bucket}/results/benchmark/pretrain_{name}/"],
                               capture_output=True)
            ckpts = sorted(glob.glob(f"{pre}/ckpt_*.pt")) or [f"{pre}/ckpt_0001150.pt"]
            job = f"abl_{name}"
            self.step(f"ft:{job}", [sys.executable, "-m", "train.finetune_sql", "--config", "configs/finetune_wikisql.yaml",
                                    "--set", f"base_ckpt={ckpts[-1]}", f"out_dir=runs/bench/{job}"])
            self.upload(job)
            self.score(job, job, ["greedy"])
            if not self.dry:
                for c in glob.glob(f"{pre}/ckpt_*.pt"):
                    os.remove(c)

        self.log("=== waiting for background evaluations")
        self.reap(wait=True)
        self.state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.save()
        self.log(f"=== benchmark queue finished; failed steps: {self.state['failed'] or 'none'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--redo", default="", help="comma-separated step keys to run again")
    a = ap.parse_args()
    Queue(a.bucket, a.dry_run).run(redo=[k for k in a.redo.split(",") if k])


if __name__ == "__main__":
    main()
