"""Download FNSPID dataset from Hugging Face into Data/FNSPID."""

from __future__ import annotations

import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Zihan1004/FNSPID"
DATA_DIR = Path(__file__).resolve().parents[1] / "Data" / "FNSPID"


def download_fnspid(local_dir: Path | None = None, extract_prices: bool = True) -> Path:
    target = local_dir or DATA_DIR
    target.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID} -> {target}")
    path = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=target,
        )
    )

    zip_path = path / "Stock_price" / "full_history.zip"
    extract_dir = path / "Stock_price" / "full_history"
    if extract_prices and zip_path.exists() and not extract_dir.exists():
        print(f"Extracting {zip_path.name} ...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)
        print(f"Extracted to {extract_dir}")

    return path


if __name__ == "__main__":
    download_fnspid()
