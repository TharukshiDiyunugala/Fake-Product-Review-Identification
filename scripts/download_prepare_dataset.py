# scripts/download_prepare_dataset.py
import os
import zipfile
import argparse
from pathlib import Path

def unzip_all(raw_dir):
    for z in Path(raw_dir).glob("*.zip"):
        with zipfile.ZipFile(z, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        print("Extracted", z)

def prepare_from_txt(raw_dir, out_csv):
    import pandas as pd
    texts, labels = [], []
    for p in Path(raw_dir).rglob("*.txt"):
        txt = p.read_text(encoding='utf-8', errors='ignore').strip()
        texts.append(txt)
        # heuristic for label: folder or filename containing 'decept' -> 1 (fake), else 0 (truthful)
        part = str(p).lower()
        label = 1 if ("decept" in part or "fake" in part) else 0
        # also check for 'pos'/'neg' for sentiment if present; prefer deceptive/truthful
        labels.append(label)
    df = pd.DataFrame({"text": texts, "label": labels})
    os.makedirs(Path(out_csv).parent, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", action="store_true", help="Try download from Kaggle dataset bit")
    parser.add_argument("--kaggle-dataset", default="KaggleUsername/dataset-name", help="Kaggle dataset path (if using kaggle)")
    parser.add_argument("--raw-dir", default="data/raw", help="where raw files are")
    parser.add_argument("--out-csv", default="data/processed/dataset.csv", help="output CSV")
    args = parser.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)

    if args.kaggle:
        # requires kaggle CLI configured
        print("Attempting kaggle download...")
        os.system(f"kaggle datasets download -p {args.raw_dir} {args.kaggle_dataset} --unzip")
    unzip_all(args.raw_dir)
    prepare_from_txt(args.raw_dir, args.out_csv)
