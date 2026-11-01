"""Publish every kept checkpoint: a GitHub release in the repository's own format, and the extra Hugging Face uploads.

    python -m scripts.publish_weights github [--dry-run]
    python -m scripts.publish_weights hf --user adiitya0929 [--dry-run]

Inputs: checkpoints/, release/weights/ (copied from the bucket) and release/TinySQL-125M-WikiSQL-lr1e-4
(scripts/export_hf.py). docs/release-v1.0.md describes every file. The main two Hugging Face repos come from
scripts/upload_hf.py.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

TAG, REPO = "v1.0", "aditya0929/tinysql"
STAGE = Path("release") / f"github-{TAG}"
GITHUB_FILES = {  # name in the release: local source
    "tinysql_125m_wikisql_lr5e-05_s0.pt": "checkpoints/tinysql_125m_wikisql_lr5e-05_s0.pt",
    "tinysql_125m_base.pt": "checkpoints/tinysql_125m_base.pt",
    "tinysql_125m_pretrain_step8600_full.pt": "checkpoints/tinysql_125m_pretrain_step8600_full.pt",
    "tinysql_125m_pretrain_step8500_full.pt": "release/weights/tinysql_125m_pretrain_step8500_full.pt",
    "tinysql_125m_wikisql_lr1e-04_s0.pt": "checkpoints/tinysql_125m_wikisql.pt",
    "TinySQL-technical-report-v1.pdf": "paper/TinySQL-technical-report-v1.pdf",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def github(dry_run):
    STAGE.mkdir(parents=True, exist_ok=True)
    sums = []
    for name, src in GITHUB_FILES.items():
        dst = STAGE / name
        if not dst.exists():
            os.link(src, dst)                                              # hard link: no second copy on disk
        sums.append(f"{sha256(dst)}  {name}")
        print(f"{name}: {dst.stat().st_size / 2**20:,.1f} MiB")
    (STAGE / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8", newline="\n")
    if dry_run:
        return
    gh = ["gh", "release", "-R", REPO]
    view = subprocess.run(gh[:2] + ["view", TAG] + gh[2:] + ["--json", "assets"], capture_output=True, text=True)
    if view.returncode != 0:                                               # draft first: public only once complete
        subprocess.run(gh[:2] + ["create", TAG] + gh[2:] + ["--draft", "--target", "main", "--title",
                       "TinySQL-125M v1.0: weights and paper", "--notes-file", "docs/release-v1.0.md"], check=True)
        have = set()
    else:
        have = {a["name"] for a in json.loads(view.stdout)["assets"]}
    for p in sorted(STAGE.iterdir()):                                      # one at a time, so a failure only retries one
        if p.name not in have:
            print(f"uploading {p.name}")
            subprocess.run(gh[:2] + ["upload", TAG, str(p)] + gh[2:], check=True)
    subprocess.run(gh[:2] + ["edit", TAG] + gh[2:] + ["--draft=false", "--latest"], check=True)
    print(f"https://github.com/{REPO}/releases/tag/{TAG}")


def hf(user, dry_run):
    folder = lambda d: {p.name: p for p in sorted(Path(d).iterdir()) if p.name != "README.md"}
    jobs = [  # (repo, branch, {path in repo: local file}, model card, commit message)
        ("TinySQL-125M-base", None, {f"original/{n}": GITHUB_FILES[n] for n in
                                     ("tinysql_125m_pretrain_step8600_full.pt", "tinysql_125m_pretrain_step8500_full.pt")},
         "TinySQL-125M-base", "Add the full pretraining state at steps 8,500 and 8,600"),
        ("TinySQL-125M-WikiSQL", None, {}, "TinySQL-125M-WikiSQL", "List the other checkpoints"),
        ("TinySQL-125M-WikiSQL", "lr1e-4-seed0", folder("release/TinySQL-125M-WikiSQL-lr1e-4"),
         "TinySQL-125M-WikiSQL-lr1e-4", "Learning-rate sweep run: LR 1e-4, seed 0"),
        ("SmolLM2-135M-WikiSQL-baseline", None, folder("release/weights/smollm2-135m"),
         "SmolLM2-135M-WikiSQL-baseline", "SmolLM2-135M fine-tuned with the TinySQL recipe"),
        ("SmolLM2-360M-WikiSQL-baseline", None, folder("release/weights/smollm2-360m"),
         "SmolLM2-360M-WikiSQL-baseline", "SmolLM2-360M fine-tuned with the TinySQL recipe"),
    ]
    for name, branch, files, card, message in jobs:
        text = (Path("hf") / card / "README.md").read_text(encoding="utf-8").replace("HF_USER", user)
        assert "EXAMPLE_OUTPUT" not in text, f"fill in the example output in hf/{card}/README.md"
        print(f"{user}/{name}" + (f"@{branch}" if branch else "") + f": {sorted(files)} + README.md")
        if dry_run:
            continue
        from huggingface_hub import CommitOperationAdd, HfApi
        api, repo = HfApi(), f"{user}/{name}"
        api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
        if branch:
            api.create_branch(repo, branch=branch, exist_ok=True)
        ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(f)) for p, f in files.items()]
        ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=text.encode("utf-8")))
        api.create_commit(repo, ops, commit_message=message, revision=branch)
        print(f"uploaded https://huggingface.co/{repo}" + (f"/tree/{branch}" if branch else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["github", "hf"])
    ap.add_argument("--user", help="Hugging Face user (hf only)")
    ap.add_argument("--dry-run", action="store_true", help="stage and check the files, upload nothing")
    a = ap.parse_args()
    if a.target == "github":
        github(a.dry_run)
    else:
        hf(a.user, a.dry_run)


if __name__ == "__main__":
    main()
