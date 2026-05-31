from datasets import load_dataset
from pathlib import Path
import json

out_dir = Path("data/raw/qasper")
out_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset("allenai/qasper")

for split, dataset in ds.items():
    out_path = out_dir / f"{split}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {len(dataset)} rows -> {out_path}")
