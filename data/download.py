"""Download the raw corpora into data/raw/<name>. Downloading is the only job of this file.

    python -m data.download --dry-run      # list the files and sizes that would be fetched
    python -m data.download                # download everything
    python -m data.download --only stack_sql wikisql
"""
import argparse
import fnmatch
import os
import tarfile
import urllib.request

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from huggingface_hub import HfApi, snapshot_download

RAW = os.path.join("data", "raw")

# patterns are fnmatch globs relative to the repo root
HF_SOURCES = {
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", ["sample/10BT/00[0-4]_00000.parquet"]),
    "stack_sql": ("bigcode/the-stack-dedup", ["data/sql/data-*-of-*.parquet"]),
    "stackexchange": ("HuggingFaceTB/stackexchange_2025_md",
                      ["dba.stackexchange.com/*", "datascience.stackexchange.com/*", "README.md"]),
    "gretel_sql": ("gretelai/synthetic_text_to_sql", ["*.parquet", "README.md"]),
    "smoltalk": ("HuggingFaceTB/smoltalk",
                 ["data/smol-magpie-ultra/train-0000[0-1]-*", "data/smol-constraints/*", "data/smol-rewrite/*",
                  "data/smol-summarize/*", "data/systemchats-30k/*", "README.md"]),
}
WIKISQL_URL = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"


def matching_files(repo, patterns):
    info = HfApi().dataset_info(repo, files_metadata=True)
    return [(s.rfilename, s.size or 0) for s in info.siblings if any(fnmatch.fnmatch(s.rfilename, p) for p in patterns)]


def download_wikisql():
    out_dir = os.path.join(RAW, "wikisql")
    os.makedirs(out_dir, exist_ok=True)
    archive = os.path.join(out_dir, "data.tar.bz2")
    if not os.path.exists(archive):
        urllib.request.urlretrieve(WIKISQL_URL, archive)
    extract_dir = os.path.join(out_dir, "extracted")           # untrusted archive -> its own empty directory
    if not os.path.isdir(extract_dir):
        os.makedirs(extract_dir)
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(extract_dir, filter="data")
    print("wikisql:", sorted(os.listdir(extract_dir)), f"({os.path.getsize(archive)/1e6:.1f} MB archive)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", help="subset of: " + ", ".join(list(HF_SOURCES) + ["wikisql"]))
    args = ap.parse_args()
    chosen = args.only or list(HF_SOURCES) + ["wikisql"]

    total = 0
    for name in chosen:
        if name == "wikisql":
            if args.dry_run:
                print(f"{'wikisql':14s} 1 file (GitHub archive, ~0.03 GB)")
            else:
                download_wikisql()
            continue
        repo, patterns = HF_SOURCES[name]
        files = matching_files(repo, patterns)
        size = sum(s for _, s in files)
        total += size
        print(f"{name:14s} {len(files):3d} files  {size/1e9:6.2f} GB   {repo}")
        if args.dry_run:
            for f, s in files[:4]:
                print(f"      {f}  ({s/1e6:.0f} MB)")
            continue
        snapshot_download(repo, repo_type="dataset", allow_patterns=patterns,
                          local_dir=os.path.join(RAW, name), max_workers=4)
        print(f"{name}: done")
    print(f"total from Hugging Face: {total/1e9:.2f} GB")


if __name__ == "__main__":
    main()
