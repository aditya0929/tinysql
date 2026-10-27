"""Upload the exported models to the Hugging Face Hub, with their model cards.

    python -m scripts.upload_hf --user YOUR_HF_USERNAME [--dry-run]
"""
import argparse
import shutil
from pathlib import Path

NAMES = ["TinySQL-125M-base", "TinySQL-125M-WikiSQL"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--dry-run", action="store_true", help="prepare the folders and print what would be uploaded")
    a = ap.parse_args()
    for name in NAMES:
        folder = Path("release") / name
        assert (folder / "model.safetensors").exists(), f"export {name} first (scripts/export_hf.py)"
        card = Path("hf") / name / "README.md"
        text = card.read_text(encoding="utf-8").replace("HF_USER", a.user)
        assert "EXAMPLE_OUTPUT" not in text, f"fill in the example output in {card}"
        (folder / "README.md").write_text(text, encoding="utf-8")
        files = sorted(p.name for p in folder.iterdir())
        print(f"{a.user}/{name}: {files}")
        if a.dry_run:
            continue
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(f"{a.user}/{name}", repo_type="model", exist_ok=True, private=False)
        api.upload_folder(repo_id=f"{a.user}/{name}", folder_path=str(folder), repo_type="model",
                          commit_message="TinySQL release v1.0 (paper DOI 10.5281/zenodo.23396770)")
        print(f"uploaded https://huggingface.co/{a.user}/{name}")


if __name__ == "__main__":
    main()
